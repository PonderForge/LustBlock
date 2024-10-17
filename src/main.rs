use std::{io::Cursor, path::{Path, PathBuf}, str::FromStr, time::Instant, process::Command, cfg, env};
use bytes::BytesMut;
use clap::Args;
use http_body_util::{BodyExt, Full};
use http_mitm_proxy::{DefaultClient, MitmProxy};
use hyper::header::{HeaderValue, CONTENT_LENGTH, CONTENT_TYPE};
use hyper::Response;
use hyper::body::Bytes;
use imageproc::{drawing::draw_filled_rect_mut, rect::Rect};
use ort::{inputs, CPUExecutionProvider, CUDAExecutionProvider, GraphOptimizationLevel, Session, SessionOutputs, TensorRTExecutionProvider};
use tracing_subscriber::EnvFilter;
use image::{ColorType, DynamicImage, GenericImageView, Rgba};
use fast_image_resize::{CpuExtensions, ResizeAlg, ResizeOptions, Resizer};
use ndarray::{s, Array, Axis, IxDyn};
use serde_json::Value;
use dirs_next;

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
    mode: bool,
    ip: String,
    port: u16,
    threasholds: Threasholds,
    optimizations: Optimizations
}

//Threasholds for each of the NSFW classes
struct Threasholds {
    porn: f64,
    sexy: f64,
    hentai: f64,
    humans: f64
}

struct Optimizations {
    cuda: bool,
    tensorrt: bool,
    threads: usize,
    avx: bool
}


#[tokio::main]
async fn main() {

    println!("Initializing LustBlock...");
    let execpath = format!("{}/",env::current_exe().unwrap().parent().unwrap().to_str().unwrap().to_owned()).leak();
    let execdir: &str = if !cfg!(debug_assertions) {execpath} else {"./"} ;
    let configdir = format!("{}/LustBlock/", dirs_next::config_dir().unwrap().to_str().unwrap());
    std::fs::create_dir_all(&configdir).unwrap();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    //Read Config File
    let config_file: String = std::fs::read_to_string(Path::new(format!("{}config.json", execdir).as_str())).unwrap();
    let v: Value = serde_json::from_str(&config_file).unwrap();
    let mode: bool = if String::from_str(v["mode"].as_str().unwrap()).unwrap() == "client"{true} else {false};
    if mode == true && !cfg!(target_os = "windows") {
        panic!("Client mode is only supported on Windows! Please set 'mode' to server in config.json.");
    }
    let config: Config = Config {
        mode: mode,
        //IP Address of the Proxy Server
        ip: if mode {String::from_str("127.0.0.1").unwrap()} else {String::from_str(v["server"]["ip"].as_str().unwrap_or("127.0.0.1")).unwrap()},
        //Port of the Proxy Server
        port: if mode {3003} else {v["server"]["port"].as_u64().unwrap_or(3003) as u16},
        //Prediction Threasholds
        threasholds: Threasholds {
            porn: v["detect_threasholds"]["porn"].as_f64().unwrap(),
            sexy: v["detect_threasholds"]["sexy"].as_f64().unwrap(),
            hentai: v["detect_threasholds"]["hentai"].as_f64().unwrap(),
            humans: v["detect_threasholds"]["humans"].as_f64().unwrap(),
        },
        optimizations: Optimizations {
            //Number of Threads to use 
            threads: v["optimizations"]["threads"].as_u64().unwrap() as usize,
            //Use the CUDA Execution Provider?
            cuda: v["optimizations"]["cuda"].as_bool().unwrap(),
            //Use AVX instuction for Resize?
            avx: v["optimizations"]["avx"].as_bool().unwrap(),
            //Use the TensorRT Execution Provider?
            tensorrt: v["optimizations"]["tensorrt"].as_bool().unwrap(),
        }
    };
    let exec_providers = [
        if config.optimizations.cuda {
            CUDAExecutionProvider::default().build()
        } else if config.optimizations.tensorrt {
            TensorRTExecutionProvider::default().build()
        } else {
            CPUExecutionProvider::default().build()
        }
    ];
    //Initialize Onnx Runtime
    let ort_init = ort::init()
		.with_execution_providers(exec_providers)
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
        println!("Creating new Certficate...");
        make_root_cert(&configdir, &config)
    };

    let proxy = MitmProxy::new(
        // This is the root cert that will be used to sign the fake certificates
        Some(root_cert),
        // This is the main Session for the NSFW detector
        Session::builder().unwrap().with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
        .with_intra_threads(config.optimizations.threads/2).unwrap().commit_from_file(format!("{}vits-classifier.onnx", execdir).as_str()).unwrap(),
        Session::builder().unwrap().with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
        .with_intra_threads(config.optimizations.threads/4).unwrap().commit_from_file(format!("{}detector.onnx", execdir).as_str()).unwrap(),
        execdir
    );
    let client = DefaultClient::new(
        tokio_native_tls::native_tls::TlsConnector::builder()
            // You must set ALPN if you want to support HTTP/2
            .request_alpns(&["h2", "http/1.1"])
            .build()
            .unwrap(),
    );
    let server = proxy
        .bind((config.ip.clone(), config.port), move |_client_addr, req, classifier, detector, execdir| {
            let client = client.clone();
            async move {
                let uri = req.uri().clone();

                // TODO: Check what the user is sending out... 

                let (res, _upgrade) = client.send_request(req).await?;

                println!("{} -> {}", uri.host().unwrap(), res.status());
                let default_content_type: HeaderValue = HeaderValue::from_str("application/octet-stream").unwrap();
                //Check if the data is a image via Content-Type Header
                let content_type = res.headers().get(CONTENT_TYPE).unwrap_or_else(||{&default_content_type}).clone();
                //Grab original response HTTP version for Spoofing
                let http_v = res.version().clone();
                //Convert Body Stream into bytes
                let (mut parts, mut data) = res.into_parts();
                let mut body = BytesMut::new();
                while let Some(Ok(chunk)) = data.frame().await {
                    body.extend(chunk.into_data().unwrap());
                }
                if content_type == "image/jpeg" || content_type == "image/png" || content_type == "image/webp" {
                    println!("Image Detected");
                    let now = Instant::now();

                    //Process the img
                    let mut input_img = image::load_from_memory(&body).unwrap();
                    let after_decoding = now.elapsed();
                    println!("  after_decoding: {:?}", after_decoding);
                    let replace_w: u32 = input_img.width().clone();
                    let replace_h: u32 = input_img.height().clone();
                    let replace_color: ColorType = input_img.color().clone();
                    if replace_h > 60 || replace_w > 60 {

                        //Create new ImageResizer
                        let mut resizer = Resizer::new();
                        let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(fast_image_resize::FilterType::Bilinear));
                        if config.optimizations.avx {
                            #[cfg(target_arch = "x86_64")]
                            unsafe {
                                resizer.set_cpu_extensions(CpuExtensions::Avx2);
                            }
                        }
                        let after_resize_create = now.elapsed() - after_decoding;
                        println!("  after_resize_create: {:?}", after_resize_create);
                        //Resize Input Image for Human Detection
                        let mut img = DynamicImage::new(640, 640, input_img.color().clone());
                        let _ = resizer.resize(&input_img, &mut img, &options);
                        //Grab Width and Height Scaling to Apply Appropiate Covering
                        let resize_w_scale: f32 = replace_w as f32/img.width().clone() as f32;
                        let resize_h_scale: f32 = replace_h as f32/img.height().clone() as f32;
                        //Convert Image to a Vector
                        let mut input = Array::zeros((1, 3, 640, 640));
                        for pixel in img.pixels() {
                            let x = pixel.0 as usize;
                            let y = pixel.1 as usize;
                            let [r, g, b, _] = pixel.2.0;
                            input[[0, 0, y, x]] = (r as f32) / 255.;
                            input[[0, 1, y, x]] = (g as f32) / 255.;
                            input[[0, 2, y, x]] = (b as f32) / 255.;
                        }
                        let after_resize_unlock = now.elapsed() - after_decoding - after_resize_create;
                        println!("  after_resize: {:?}", after_resize_unlock);
                        //Run the Human Detector (YOLOv11) on the Image Vector
                        let output_tensor: SessionOutputs = detector.run(inputs!["images" => input.view()].unwrap()).unwrap();
                        let outputs = output_tensor["output0"].try_extract_tensor::<f32>().unwrap().view().t().to_owned();
                        let after_det = now.elapsed() - after_resize_unlock - after_decoding - after_resize_create;
                        println!("  after_det: {:?}", after_det);
                        //Run post processing on Human Detector into Vector
                        let boxes = process_yolo_output(outputs, 640, 640);
                        let after_det_process = now.elapsed() - after_det - after_resize_unlock - after_decoding - after_resize_create;
                        println!("  after_det_process: {:?}", after_det_process);
                        let mut replace_image = false;
                        for i in &boxes {
                            if i.4 > config.threasholds.humans as f32 {
                                //Run NSFW Classification on All Humans Detected
                                let metrix = classify_image(&classifier, &img.crop(i.0 as u32, i.1 as u32, (i.2-i.0).abs() as u32, (i.3-i.1).abs() as u32), &mut resizer, &options);
                                println!("  Human Metrics: {:?}", metrix);
                                if metrix[1] > config.threasholds.hentai as f32 || metrix[3] > config.threasholds.porn as f32 || metrix[4] > config.threasholds.sexy as f32 {
                                    //Cover Human if NSFW
                                    draw_filled_rect_mut(&mut input_img, Rect::at((i.0*resize_w_scale) as i32, (i.1*resize_h_scale) as i32).of_size(((i.2-i.0).abs() * resize_w_scale) as u32, ((i.3-i.1).abs() * resize_h_scale) as u32), Rgba([(255f32*metrix[1]) as u8, (255f32*metrix[3]) as u8, (255f32*metrix[4]) as u8, 255u8]));
                                    replace_image = true;
                                }
                            }
                        }
                        if replace_image  {
                            let after_class_process = now.elapsed() - after_det_process - after_det - after_resize_unlock - after_decoding - after_resize_create;
                            println!("  after_class_process: {:?}", after_class_process);
                            //Replace Orginal Image with Scrubbed Image
                            println!("  Image is NSFW: Edited");
                            let mut bytes: Vec<u8> = Vec::new();
                            input_img.write_to(&mut Cursor::new(&mut bytes), if content_type == "image/png" {image::ImageFormat::Png} else if content_type == "image/webp" {image::ImageFormat::WebP} else {image::ImageFormat::Jpeg}).unwrap();
                            body = BytesMut::from(bytes.as_slice());
                            parts.headers.insert(CONTENT_LENGTH, body.len().into());
                            let after_image_replace = now.elapsed() - after_class_process - after_det_process - after_det - after_resize_unlock - after_decoding - after_resize_create;
                            println!("  after_image_replace: {:?}", after_image_replace);
                        } else {
                            println!("  Image is OK: Allowed");
                        }
                        //If the Human Detector does not find any humans, then Classifier runs on Whole Image
                        if boxes.is_empty() {
                            let metrix = classify_image(&classifier, &img, &mut resizer, &options);
                            let after_class_process = now.elapsed() - after_det_process - after_det - after_resize_unlock - after_decoding - after_resize_create;
                            println!("  after_class_process: {:?}", after_class_process);
                            println!("  Overall Metrics: {:?}", metrix);
                            if metrix[1] > config.threasholds.hentai as f32 || metrix[3] > config.threasholds.porn as f32 || metrix[4] > config.threasholds.sexy as f32 {
                                //Replace Whole Image with Distraction
                                let mut replacement = DynamicImage::new(replace_w, replace_h, replace_color);
                                let _ = resizer.resize(&image::open(Path::new(format!("{}distraction.jpg", execdir).as_str())).unwrap(), &mut replacement, &options);
                                let mut bytes: Vec<u8> = Vec::new();
                                replacement.write_to(&mut Cursor::new(&mut bytes), if content_type == "image/png" {image::ImageFormat::Png} else if content_type == "image/webp" {image::ImageFormat::WebP} else {image::ImageFormat::Jpeg}).unwrap();
                                body = BytesMut::from(bytes.as_slice());
                                println!("  Image is NSFW: Blocked");
                                let after_image_replace = now.elapsed() - after_class_process - after_det_process - after_det - after_resize_unlock - after_decoding - after_resize_create;
                                println!("  after_image_replace: {:?}", after_image_replace);
                            } else {
                                println!("  Image is OK: Allowed")
                            }
                        } else {
                            if !replace_image  {
                                let after_class_process = now.elapsed() - after_det_process - after_det - after_resize_unlock - after_decoding - after_resize_create;
                                println!("  after_class_process: {:?}", after_class_process);
                            }
                        }
                        println!("  Time: {:?}", now.elapsed());
                    } else {
                        println!("  Image is Too Small: Allowed");
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

    println!("HTTPS Proxy is listening on http://{}:{}", &config.ip, config.port);

    println!();

    if config.mode {
        #[cfg(target_family = "windows")]
        {
            use windows_registry::CURRENT_USER;
            let proxy_set = CURRENT_USER.create("SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Internet Settings").unwrap();
            let _ = proxy_set.set_string("ProxyServer", &format!("{}:{}", &config.ip, config.port));
            let _ = proxy_set.set_u32("ProxyEnable", 1);
            println!("Proxy added to Computer Network Stack");
            println!("Enjoy a (sexual) temptation-less internet!");
        }
    } else {
        if !cfg!(target_os = "windows") {
            println!("Trust the pub.crt certificate if you want to use HTTPS on the device");
        }
        println!("Use the URL above in your proxy settings to apply ");
        println!("Enjoy a (sexual) temptation-less internet!");
    }

    /*
        Debug Test if you wish to not change your proxy settings in Windows
        curl -i https://news.clark.edu/wp-content/uploads/2024/10/20241003-8P5A5738-1-1024x683.jpg --proxy 127.0.0.1:3003 --cacert C:\Users\{Username}\AppData\Roaming\LustBlock\pub.crt --ssl-revoke-best-effort
    */
    #[cfg(target_os = "windows")]
    if config.mode {
        unsafe{ register_close_handler(); }
    }
    server.await;

}

#[cfg(target_os = "windows")]
#[link(name = "close", kind = "static")]
extern "C" {
    fn register_close_handler();
}

fn make_root_cert(configdir: &String, config: &Config) -> rcgen::CertifiedKey {
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
    if config.mode {
        if cfg!(target_os = "windows") { 
            println!("Adding to Root Certificate Store");
            println!("{:?}", Command::new("cmd")
            .args(["/C", format!("certutil -user -addstore Root {}pub.crt", configdir).as_str()])
            .output()
            .expect("failed to execute process"));
        }
    }
    rcgen::CertifiedKey { cert, key_pair }
}
fn classify_image (model: &Session, image: &DynamicImage, resizer: &mut Resizer, options: &ResizeOptions) -> Vec<f32> {
    //Reformat Image to Classifer Model Input
    let mut img = DynamicImage::new(224, 224, image.color().clone());
    let _ = resizer.resize(image, &mut img, options);
    let mut input = Array::zeros((1, 3, 224, 224));
    for pixel in img.pixels() {
        let x = pixel.0 as usize;
        let y = pixel.1 as usize;
        let [r, g, b, _] = pixel.2.0;
        input[[0, 0, y, x]] = (r as f32) / 255.;
        input[[0, 1, y, x]] = (g as f32) / 255.;
        input[[0, 2, y, x]] = (b as f32) / 255.;
    }
    //Perform Inference
    let output_tensor: SessionOutputs = model.run(inputs!["pixel_values" => input.view()].unwrap()).unwrap();
    let outputs = output_tensor["logits"].try_extract_tensor::<f32>().unwrap();
    let mut metrix: Vec<f32> = Vec::new();
    for output in outputs.rows() {
        metrix = output.to_vec();
    }
    return metrix;
}
// Function used to convert RAW output from YOLOv11 to an array
// Returns array of detected objects in a format [(x1,y1,x2,y2,object_type,probability),..]
fn process_yolo_output(output:Array<f32,IxDyn>,img_width: u32, img_height: u32) -> Vec<(f32,f32,f32,f32, f32)> {
    let mut boxes = Vec::new();
    let output = output.slice(s![..,..,0]);
    for row in output.axis_iter(Axis(0)) {
        let row:Vec<_> = row.iter().map(|x| *x).collect();
        let (class_id, prob) = row.iter().skip(4).enumerate()
            .map(|(index,value)| (index,*value))
            .reduce(|accum, row| if row.1>accum.1 { row } else {accum}).unwrap();
        if class_id == 0 {
            if prob < 0.5 {
                continue
            }
            let xc = row[0]/640.0*(img_width as f32);
            let yc = row[1]/640.0*(img_height as f32);
            let w = row[2]/640.0*(img_width as f32);
            let h = row[3]/640.0*(img_height as f32);
            let x1 = xc - w/2.0;
            let x2 = xc + w/2.0;
            let y1 = yc - h/2.0;
            let y2 = yc + h/2.0;
            boxes.push((x1,y1,x2,y2,prob));
        }
    }

    let mut result = Vec::new();
    while boxes.len()>0 {
        result.push(boxes[0]);
        boxes = boxes.iter().filter(|box1| iou(&boxes[0],box1) < 0.7).map(|x| *x).collect()
    }
    return result;
}

// Function calculates "Intersection-over-union" coefficient for specified two boxes
// Returns Intersection over union ratio as a float number
fn iou(box1: &(f32, f32, f32, f32, f32), box2: &(f32, f32, f32, f32, f32)) -> f32 {
    return intersection(box1, box2) / union(box1, box2);
}

// Function calculates union area of two boxes
// Returns Area of the boxes union as a float number
fn union(box1: &(f32, f32, f32, f32, f32), box2: &(f32, f32, f32, f32, f32)) -> f32 {
    let (box1_x1,box1_y1,box1_x2,box1_y2,_) = *box1;
    let (box2_x1,box2_y1,box2_x2,box2_y2,_) = *box2;
    let box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1);
    let box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1);
    return box1_area + box2_area - intersection(box1, box2);
}

// Function calculates intersection area of two boxes
// Returns Area of intersection of the boxes as a float number
fn intersection(box1: &(f32, f32, f32, f32, f32), box2: &(f32, f32, f32, f32, f32)) -> f32 {
    let (box1_x1,box1_y1,box1_x2,box1_y2,_) = *box1;
    let (box2_x1,box2_y1,box2_x2,box2_y2,_) = *box2;
    let x1 = box1_x1.max(box2_x1);
    let y1 = box1_y1.max(box2_y1);
    let x2 = box1_x2.min(box2_x2);
    let y2 = box1_y2.min(box2_y2);
    return (x2-x1)*(y2-y1);
}