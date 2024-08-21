# LustBlock
 A 93% accurate, ONNX-based, NSFW image proxy, for those who want to do the right thing
![Logo and replacement image for the proxy](https://raw.githubusercontent.com/PonderForge/LustBlock/main/distraction.jpg)
# Install
If you want to use it, check out the releases first, we compile for CUDA 12 and CPU on both Linux and Windows.
1. Go to our latest release and download it.
- On Linux, use: `wget https://github.com/PonderForge/LustBlock/releases/download/v{version}/{os}-{execution provider}.tar.gz -o lustblock.tar.gz`
- On Windows, use: `curl https://github.com/PonderForge/LustBlock/releases/download/v{version}/{os}-{execution provider}.zip -o lustblock.tar.gz`\
(Replace the brackets with what your machine uses)
2. Unzip/Extract it using your favorite extractor
- On Linux, it looks like this: `tar -xvzf lustblock.tar.gz`
- On Windows, I suggest using the 7z graphical interface.
3. Change into the release directory and run the main program
You can run it using `./lustblock` on Linux or `lustblock.exe` on Windows.
4. Configure your operating system to use the proxy, via the server's local or public IP:
- On Ubuntu, follow [this tutorial](https://phoenixnap.com/kb/ubuntu-proxy-settings)
- On Windows, follow [this tutorial](https://support.microsoft.com/en-us/windows/use-a-proxy-server-in-windows-03096c53-0554-4ffe-b6ab-8b1deee8dae1)
- On Android, follow [this tutorial](https://proxyway.com/guides/android-proxy-settings)
- On iOS, follow [this tutorial](https://libertyshield.kayako.com/article/32-manual-proxy-ios-iphone-and-ipad)
5. Finally, to use HTTPS and not make your browser lose it, install the fake cert:
- On Ubuntu, follow [this tutorial](https://askubuntu.com/questions/73287/how-do-i-install-a-root-certificate/94861#94861)
- On Windows, follow [this tutorial](https://web.archive.org/web/20160612045445/http://windows.microsoft.com/en-ca/windows/import-export-certificates-private-keys#1TC=windows-7)
- On Android, follow [this tutorial](http://wiki.cacert.org/FAQ/ImportRootCert#Android_Phones_.26_Tablets)
- On iOS, follow [this tutorial](http://jasdev.me/intercepting-ios-traffic)
# Compiling
If you want to compile with more of ONNX's Execution providers such as TensorRT, or you just want to change the default thresholds for removing NSFW, follow this tutorial.
1. Well First off grab the directory By downloading the zip, GitHub desktop, or running:\
`git clone https://github.com/PonderForge/LustBlock.git && cd LustBlock`\
2. Next use Cargo to build via:\
`cargo build`
3. Follow the installing section from step 3.
# Configuration 
## Execution Providers
LustBlock uses the ORT, an ONNX library for Rust, which provides many different execution providers, or modules that make the AI run faster. 
You can add one by:
1. Finding the Execution Provider and if ORT supports it, [here](https://ort.pyke.io/perf/execution-providers).
2. If ORT does support it, you have to add the Cargo feature in the Cargo.toml
3. Then include it in the ort "use" statement in src/main.rs:
`use ort::{inputs, {put the provider name here}, GraphOptimizationLevel, Session, SessionOutputs, CPUExecutionProvider};`
More on the list of Execution provider structs [here](https://docs.rs/ort/2.0.0-rc.2/ort/index.html?search=ExecutionProvider)
4. Finally add it to the ort::init in src/main.rs like so:
`.with_execution_providers([{put the provider name here}::default().build(), CPUExecutionProvider::default().build()])`
5. You may need to install the required libraries for the execution provider, like with CUDA you need CUDA 12 and cuDNN 9, more on this [here](https://ort.pyke.io/perf/execution-providers).
## Thresholds
So AIs are not completely accurate (our is 93%), and ORT only returns the likelihood of the image being a certain classifier. This means that we see if an image is NSFW if the likelihood of the image being Hentai, Porn, or Sexy is high enough via the set defaults in src/main.rs. You can change these to your liking as long as you know that an increase in threshold could increase how much NSFW slips through, and a decrease could increase neutral images getting redacted. Our thresholds have been set from my data, but you may have different results or standards. You can find them in src/main.rs at HENTAI_DETECT_THREASHOLD, PORN_DETECT_THREASHOLD, and SEXY_DETECT_THREASHOLD as compile time constants.
# Why?
Since you are this far into the README, you must be interested in the motivation for this project.
Lust by definition is "very strong sexual desire" for another person. This desire usually comes from our heart and is amplified by hormones. Left unchecked, it can destroy everything we know and love. Porn and Sexual images are one of the worst things to lust after. It is cheap, hideable, and spreads like wildfire. A simple Google search could bring up results that start you on the path. Which is why we are here. Thanks to the help of GantMan's NSFW models, we can quickly detect and remove NSFW images from the web before it reaches you and your loved ones. Did you know that when one starts to see porn on purpose, their chance of getting a divorce rate [doubles](https://www.science.org/content/article/divorce-rates-double-when-people-start-watching-porn)? If you have not had to fight lust, you are very lucky. Let me tell you that even without getting into Porn, just the sexy images, it takes suffering, and will to break free. It is literally the [internet cocaine](https://www.provenmen.org/porn-damages-brain/). Fight it, protect others and yourself from it, save those who are in it. Win. We were not created for this. Yet here we are. Hmmm, there was a book that warned us about that. Oh, yeah. It was the [Bible](https://www.bible.com/).
# Credits
- [Gantman/nsfw_model](https://github.com/GantMan/nsfw_model): He did what no one else had the nerve to do. I didn't want to even touch the data, so thanks for not making me have to!
- [hatoo/http-mitm-proxy](https://github.com/hatoo/http-mitm-proxy/tree/master): I modified his library for the HTTPS proxy! Couldn't have done it without this project.
- [pykeio/ort](https://github.com/pykeio/ort): Literally, most helpful project.
- My mom: Taught me how to fight lust, how to see the better way, and how to be a better man. Oh and put up with a verbally abusive man for 26 years to keep the family together. Love you!
- Jesus Christ: My savior, my redeemer, my rock, my king, my commander, and literally the sole reason I exist. Wrote the book that warned us about lust and still loves us when we ignore it. All Hail King Jesus!\
\
This was created by PonderForge, if you use this code, give credit where credit is due.\
Pslam 111:2 "Great are the works of the LORD; they are pondered by all who delight in them."
