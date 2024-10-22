use bytes::Bytes;
use data::Data;
use http_body_util::{combinators::BoxBody, BodyExt, Empty};
use hyper::{
    body::{Body, Incoming},
    server,
    service::service_fn,
    Method, Request, Response, StatusCode,
};
use hyper_util::rt::{TokioExecutor, TokioIo};
use std::{borrow::Borrow, future::Future, net::SocketAddr, sync::Arc};
use tls::server_config;
use tokio::net::{TcpListener, ToSocketAddrs};
use ort::Session;
use yaml_rust2::yaml::Hash;

pub use futures;
pub use hyper;
pub use tokio_native_tls;
pub mod data;
pub mod default_client;
mod tls;

pub use default_client::DefaultClient;

#[derive(Clone)]
/// The main struct to run proxy server
pub struct MitmProxy<C> {
    /// Root certificate to sign fake certificates. You may need to trust this certificate on client application to use HTTPS.
    ///
    /// If None, proxy will just tunnel HTTPS traffic and will not observe HTTPS traffic.
    pub root_cert: Option<C>,
    pub classifier: Data<Session>,
    pub detector: Data<Session>,
    pub site_reactions: Data<Hash>, 
    pub execdir: &'static str
}

impl<C> MitmProxy<C> {
    /// Create a new MitmProxy
    pub fn new(root_cert: Option<C>, classifier: Session, detector: Session, site_reactions: Hash, execdir: &'static str) -> Self {
        Self { root_cert, classifier: Data::new(classifier), detector: Data::new(detector), site_reactions: Data::new(site_reactions), execdir: execdir}
    }
}

// pub type Handler<B, E> = Fn(Request<Incoming>) -> Result<Response<B>, E>;

impl<C: Borrow<rcgen::CertifiedKey> + Send + Sync + 'static> MitmProxy<C> {
    /// Bind to a socket address and return a future that runs the proxy server.
    /// URL for requests that passed to service are full URL including scheme.
    pub async fn bind<A: ToSocketAddrs, S, B, E, E2, F>(
        self,
        addr: A,
        service: S,
    ) -> Result<impl Future<Output = ()>, std::io::Error>
    where
        B: Body<Data = Bytes, Error = E> + Send + Sync + 'static,
        E: std::error::Error + Send + Sync + 'static,
        E2: std::error::Error + Send + Sync + 'static,
        S: Fn(SocketAddr, Request<Incoming>, Data<Session>, Data<Session>, Data<Hash>, &'static str) -> F + Send + Sync + Clone + 'static,
        F: Future<Output = Result<Response<B>, E2>> + Send,
    {
        let listener = TcpListener::bind(addr).await?;

        let proxy = Arc::new(self);
        Ok(async move {
            loop {
                let Ok((stream, client_addr)) = listener.accept().await else {
                    continue;
                };  

                let service = service.clone();
                let proxy = proxy.clone();
                tokio::spawn(async move {

                    if let Err(err) = server::conn::http1::Builder::new()
                        .preserve_header_case(true)
                        .title_case_headers(true)
                        .serve_connection(
                            TokioIo::new(stream),
                            service_fn(|req| {
                                Self::proxy(proxy.clone(), client_addr, req, service.clone())
                            }),
                        )
                        .with_upgrades()
                        .await
                    {
                        tracing::error!("Error in proxy: {}", err);
                    }
                });
            }
        })
    }

    async fn proxy<S, B, E, E2, F>(
        proxy: Arc<MitmProxy<C>>,
        client_addr: SocketAddr,
        req: Request<Incoming>,
        service: S,
    ) -> Result<Response<BoxBody<Bytes, E>>, E2>
    where
        S: Fn(SocketAddr, Request<Incoming>, Data<Session>, Data<Session>, Data<Hash>, &'static str) -> F + Send + Clone + 'static,
        F: Future<Output = Result<Response<B>, E2>> + Send,
        B: Body<Data = Bytes, Error = E> + Send + Sync + 'static,
        E: std::error::Error + Send + Sync + 'static,
        E2: std::error::Error + Send + Sync + 'static,
    {
        if req.method() == Method::CONNECT {
            // https
            let Some(connect_authority) = req.uri().authority().cloned() else {
                tracing::error!(
                    "Bad CONNECT request: {}, Reason: Invalid Authority",
                    req.uri()
                );
                return Ok(no_body(StatusCode::BAD_REQUEST));
            };

            tokio::spawn(async move {
                let Ok(client) = hyper::upgrade::on(req).await else {
                    tracing::error!(
                        "Bad CONNECT request: {}, Reason: Invalid Upgrade",
                        connect_authority
                    );
                    return;
                };
                if let Some(root_cert) = proxy.root_cert.as_ref() {
                    let Ok(server_config) =
                        // Even if URL is modified by middleman, we should sign with original host name to communicate client.
                        server_config(connect_authority.host().to_string(), root_cert.borrow(), true)
                    else {
                        tracing::error!("Failed to create server config for {}", connect_authority.host());
                        return;
                    };
                    // TODO: Cache server_config
                    let server_config = Arc::new(server_config);
                    let tls_acceptor = tokio_rustls::TlsAcceptor::from(server_config);
                    let client = match tls_acceptor.accept(TokioIo::new(client)).await {
                        Ok(client) => client,
                        Err(err) => {
                            tracing::error!(
                                "Failed to accept TLS connection for {}, {}",
                                connect_authority.host(),
                                err
                            );
                            return;
                        }
                    };
                    let classifier = proxy.classifier.clone();
                    let detector = proxy.detector.clone();
                    let execdir = proxy.execdir;
                    let site_reactions = proxy.site_reactions.clone();
                    let f = move |mut req: Request<_>| {
                        let connect_authority = connect_authority.clone();
                        let service = service.clone();
                        let classi2 = classifier.clone();
                        let detect2 = detector.clone();
                        let site_reactions2 = site_reactions.clone();
                        async move {
                            inject_authority(&mut req, connect_authority.clone());
                            service(client_addr, req, classi2, detect2, site_reactions2, execdir).await
                        }
                    };
                    let res = if client.get_ref().1.alpn_protocol() == Some(b"h2") {
                        server::conn::http2::Builder::new(TokioExecutor::new())
                            .serve_connection(TokioIo::new(client), service_fn(f))
                            .await
                    } else {
                        server::conn::http1::Builder::new()
                            .preserve_header_case(true)
                            .title_case_headers(true)
                            .serve_connection(TokioIo::new(client), service_fn(f))
                            .with_upgrades()
                            .await
                    };

                    if let Err(err) = res {
                        tracing::error!("Error in proxy: {}", err);
                    }
                }
            });

            Ok(Response::new(
                http_body_util::Empty::new()
                    .map_err(|never: std::convert::Infallible| match never {})
                    .boxed(),
            ))
        } else {
            // http
            service(client_addr, req, proxy.classifier.clone(), proxy.detector.clone(), proxy.site_reactions.clone(), proxy.execdir)
                .await
                .map(|res| res.map(|b| b.boxed()))
        }
    }
}

fn no_body<E>(status: StatusCode) -> Response<BoxBody<Bytes, E>> {
    let mut res = Response::new(Empty::new().map_err(|never| match never {}).boxed());
    *res.status_mut() = status;
    res
}

fn inject_authority<B>(request_middleman: &mut Request<B>, authority: hyper::http::uri::Authority) {
    let mut parts = request_middleman.uri().clone().into_parts();
    parts.scheme = Some(hyper::http::uri::Scheme::HTTPS);
    if parts.authority.is_none() {
        parts.authority = Some(authority);
    }
    *request_middleman.uri_mut() = hyper::http::uri::Uri::from_parts(parts).unwrap();
}
