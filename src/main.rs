use std::{io::Cursor, path::{Path, PathBuf}, str::FromStr, time::Instant};
use bytes::BytesMut;
use clap::Args;
use http_body_util::{BodyExt, Full};
use http_mitm_proxy::{DefaultClient, MitmProxy};
use hyper::header::HeaderValue;
use hyper::Response;
use hyper::body::Bytes;
use ort::{inputs, CPUExecutionProvider, CUDAExecutionProvider, GraphOptimizationLevel, Session, SessionOutputs};
use tracing_subscriber::EnvFilter;
use image::{imageops::FilterType, GenericImageView};
use ndarray::Array;
use serde_json::Value;
use dirs_next::config_dir;

mod data;


//Structs for Sanity
#[derive(Args, Debug)]
struct ExternalCert {
    #[arg(required = false)]
    cert: PathBuf,
    #[arg(required = false)]
    private_key: PathBuf,
}

//Main Config from main.conf
struct Config {
    ip: String,
    port: u16,
    threasholds: Threasholds,
    threads: usize,
    cuda: bool
}

//Threasholds for each of the NSFW classes
struct Threasholds {
    porn: f64,
    sexy: f64,
    hentai: f64
}


#[tokio::main]
async fn main() {
    println!("Initializing LustBlock...");
    let configdir = format!("{}/LustBlock/", dirs_next::config_dir().unwrap().to_str().unwrap());
    std::fs::create_dir_all(&configdir).unwrap();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    //Read Config File
    let config_file = std::fs::read_to_string(Path::new("config.json")).unwrap();
    let v: Value = serde_json::from_str(&config_file).unwrap();
    let config = Config {
        ip: String::from_str(v["ip"].as_str().unwrap()).unwrap(),
        port: v["port"].as_u64().unwrap() as u16,
        threasholds: Threasholds {
            porn: v["detect_threasholds"]["porn"].as_f64().unwrap(),
            sexy: v["detect_threasholds"]["sexy"].as_f64().unwrap(),
            hentai: v["detect_threasholds"]["hentai"].as_f64().unwrap(),
        },
        threads: v["threads"].as_u64().unwrap() as usize,
        cuda: v["cuda"].as_bool().unwrap()
    };

    //Initialize Onnx Runtime
    let ort_init = ort::init()
		.with_execution_providers([if config.cuda {CUDAExecutionProvider::default().build()} else {CPUExecutionProvider::default().build()}])
		.commit();
    if ort_init.is_err() {
        panic!("ONNX was not correctly initalized! :(");
    }
    let public_key = std::fs::read_to_string(Path::new(format!("{}pub.crt", configdir).as_str()));
    let private_key = std::fs::read_to_string(Path::new(format!("{}priv.crt", configdir).as_str()));
    //Check if the HTTPS keys exist
    let root_cert = if !public_key.is_err() && !private_key.is_err() {
        // If so, Use existing key
        let param = rcgen::CertificateParams::from_ca_cert_pem(
            &public_key.unwrap(),
        )
        .unwrap();
        let key_pair = rcgen::KeyPair::from_pem(&private_key.unwrap()).unwrap();
        let cert = param.self_signed(&key_pair).unwrap();

        rcgen::CertifiedKey { cert, key_pair }
    } else {
        //Else, make and save them
        make_root_cert(&configdir)
    };

    let proxy = MitmProxy::new(
        // This is the root cert that will be used to sign the fake certificates
        Some(root_cert),
        // This is the main Session for the NSFW detector
        Session::builder().unwrap().with_optimization_level(GraphOptimizationLevel::Level1).unwrap()
        .with_inter_threads(config.threads).unwrap().commit_from_file("model.onnx").unwrap()
    );

    let client = DefaultClient::new(
        tokio_native_tls::native_tls::TlsConnector::builder()
            // You must set ALPN if you want to support HTTP/2
            .request_alpns(&["h2", "http/1.1"])
            .build()
            .unwrap(),
    );
    let server = proxy
        .bind((config.ip.clone(), config.port), move |_client_addr, req, detector| {
            let client = client.clone();
            async move {
                let uri = req.uri().clone();

                // TODO: Check what the user is sending out... 

                let (res, _upgrade) = client.send_request(req).await?;

                println!("{} -> {}", uri.host().unwrap(), res.status());
                let default_content_type: HeaderValue = HeaderValue::from_str("application/octet-stream").unwrap();
                //Check if the data is a image via Content-Type Header
                let content_type = res.headers().get("Content-Type").unwrap_or_else(||{&default_content_type}).clone();
                //Grab original response HTTP version for Spoofing
                let http_v = res.version().clone();
                //Convert Body Stream into bytes
                let (parts, mut data) = res.into_parts();
                let mut body = BytesMut::new();
                while let Some(Ok(chunk)) = data.frame().await {
                    body.extend(chunk.into_data().unwrap());
                }
                if content_type == "image/jpeg" || content_type == "image/png" || content_type == "image/webp" {
                    println!("Image Detected");
                    //Unlock the Model from the MITM Proxy threads
                    let model = detector.lock().unwrap();

                    //Process the img
                    let original_img = image::load_from_memory(&body).unwrap();
                    let replace_w = original_img.width().clone();
                    let replace_h = original_img.height().clone();
                    let img = original_img.resize_exact(299, 299, FilterType::CatmullRom);
                    let mut input = Array::zeros((1, 299, 299, 3));
                    for pixel in img.pixels() {
                        let x = pixel.0 as _;
                        let y = pixel.1 as _;
                        let [r, g, b, _] = pixel.2.0;
                        input[[0, y, x, 0]] = (r as f32) / 255.;
                        input[[0, y, x, 1]] = (g as f32) / 255.;
                        input[[0, y, x, 2]] = (b as f32) / 255.;
                    }


                    //Perform Inference
                    let now = Instant::now();
                    let output_tensor: SessionOutputs = model.run(inputs!["input_1" => input.view()].unwrap()).unwrap();
                    let outputs = output_tensor["dense_3"].try_extract_tensor::<f32>().unwrap();
                    println!("  Elapsed: {:.2?}", now.elapsed());
                    let mut metrix: Vec<_> = Vec::new();
                    for output in outputs.rows() {
                        metrix = output.to_vec();
                    }
                    println!("  Metrics: {:?}", metrix);
                    //Process Metrics
                    if metrix[1] > config.threasholds.hentai as f32 || metrix[3] > config.threasholds.porn as f32 || metrix[4] > config.threasholds.sexy as f32 {
                        let replacement = image::open(Path::new("distraction.jpg")).unwrap().resize(replace_w, replace_h, FilterType::CatmullRom);
                        let mut bytes: Vec<u8> = Vec::new();
                        replacement.write_to(&mut Cursor::new(&mut bytes), if content_type == "image/png" {image::ImageFormat::Png} else if content_type == "image/webp" {image::ImageFormat::WebP} else {image::ImageFormat::Jpeg}).unwrap();
                        body = BytesMut::from(bytes.as_slice());
                        println!("  Image is NSFW: Blocked")
                    } else {
                        println!("  Image is OK: Allowed")
                    }
                }

                //Reconstruct and return response 
                let mut after = Response::<Full<Bytes>>::from_parts(parts, Full::new(body.into()));
                *after.version_mut() = http_v;
                Ok::<_, http_mitm_proxy::default_client::Error>(after)
            }
        })
        .await
        .unwrap();

    println!("HTTP Proxy is listening on http://{}:{}", &config.ip, config.port);

    println!();
    println!("Trust the pub.crt cert if you want to use HTTPS");

    /*
        Save this cert to ca.crt and try it with curl like this:
        curl -i https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR8xF41_qUV3Kue3McuviMZmzj0FqCD7O2uEp0du0i7Hz4ZgpdJ --proxy 127.0.0.1:3003 --cacert ./pub.crt --ssl-revoke-best-effort
    */
    server.await;
}

fn make_root_cert(configdir: &String) -> rcgen::CertifiedKey {
    let mut param = rcgen::CertificateParams::default();

    param.distinguished_name = rcgen::DistinguishedName::new();
    param.distinguished_name.push(
        rcgen::DnType::CommonName,
        rcgen::DnValue::Utf8String("LustBlock MITM cert".to_string()),
    );
    param.key_usages = vec![
        rcgen::KeyUsagePurpose::KeyCertSign,
        rcgen::KeyUsagePurpose::CrlSign,
    ];
    param.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);

    let key_pair = rcgen::KeyPair::generate().unwrap();
    let cert = param.self_signed(&key_pair).unwrap();
    let _ = std::fs::write(Path::new(format!("{}pub.crt", configdir).as_str()), cert.pem());
    let _ = std::fs::write(Path::new(format!("{}priv.crt", configdir).as_str()), key_pair.serialize_pem());
    rcgen::CertifiedKey { cert, key_pair }
}