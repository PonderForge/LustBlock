# LustBlock
 A 93% accurate, ONNX-based, NSFW image proxy, to filter sexual images from your browser
![Logo and replacement image for the proxy](https://raw.githubusercontent.com/PonderForge/LustBlock/main/distraction.jpg)
# Install
## Linux
1. Go to our latest release and download it.
`wget https://github.com/PonderForge/LustBlock/releases/download/v0.0.2/linux-x64.tar.gz -o lustblock.tar.gz`
(Replace the brackets with what your machine uses)
2. Unzip/Extract it using your favorite extractor
`tar -xvzf lustblock.tar.gz`
3. Change into the release directory and run the main program
You can run it using `cd lustblock && ./lustblock`
## Windows
1. Download lustblock-setup.exe
2. Follow instructions through the prompts
3. Skip Post install if you are only installing for your system
## Post Install
1. Configure your operating system to use the proxy, via the server's local or public IP:
- On Ubuntu, follow [this tutorial](https://phoenixnap.com/kb/ubuntu-proxy-settings)
- On Windows, follow [this tutorial](https://support.microsoft.com/en-us/windows/use-a-proxy-server-in-windows-03096c53-0554-4ffe-b6ab-8b1deee8dae1)
- On Android, follow [this tutorial](https://proxyway.com/guides/android-proxy-settings)
- On iOS, follow [this tutorial](https://libertyshield.kayako.com/article/32-manual-proxy-ios-iphone-and-ipad)
2. Finally, to use HTTPS and not make your browser lose it, install the fake cert:
- On Ubuntu, follow [this tutorial](https://askubuntu.com/questions/73287/how-do-i-install-a-root-certificate/94861#94861)
- On Windows, follow [this tutorial](https://web.archive.org/web/20160612045445/http://windows.microsoft.com/en-ca/windows/import-export-certificates-private-keys#1TC=windows-7)
- On Android, follow [this tutorial](http://wiki.cacert.org/FAQ/ImportRootCert#Android_Phones_.26_Tablets)
- On iOS, follow [this tutorial](http://jasdev.me/intercepting-ios-traffic)
# Compiling
If you want to compile with more of ONNX's Execution providers such as TensorRT, or you just want to screw with the code, follow this tutorial.
1. Download the source code either via GitHub desktop, or by running:\
`git clone https://github.com/PonderForge/LustBlock.git && cd LustBlock`
2. Write any edits to the code as you wish
3. Next use Cargo to build via:\
`cargo build`
4. Go to the target/debug directory and run LustBlock
# Configuration 
We have 2 methods of configuration, either via config.json which exposes basic config, or via recompilation.
## Execution Providers
LustBlock uses the ORT, an ONNX library for Rust, which provides many different execution providers, or modules that make the AI Proxy run faster. CUDA is included by default in releases and configurable via config.json, the rest of the execution providers you must compile yourself.
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
So AIs are not completely accurate (our is 93%), and LustBlock only returns the likelihood of the image being a certain class of image. This means that an image is identified as NSFW if the likelihood of the image being Hentai, Porn, or Sexy is a number higher than the numbers as set in config.json. You can change these to your preference as long as you know that an increase in thresholds could increase how many NSFW images slip through, and a decrease in thresholds could increase neutral images getting filtered.
# Why?
Since you are this far into the README, you must be interested in the motivation for this project.
Lust by definition is "very strong sexual desire" for another person as almost everyone experiences. Left unchecked, it starts to destroy you internally which will begin to affect others around you. Porn and Sexual images are one of the easiest things to lust after because it is cheap, hideable, and spreads faster than my dog can run. Even a simple Google search could bring up results that could spiral you down a path of torment and destruction, just due to the girl or boy that was taught that they are nothing more than meat. That is why LustBlock was created. Thanks to the help of GantMan's NSFW models, we can quickly detect and remove NSFW images from the web before it reaches you and your loved ones. Did you know that when one starts to see porn on purpose, their chance of getting a divorce rate [doubles](https://www.science.org/content/article/divorce-rates-double-when-people-start-watching-porn)? If you have not had to fight lust, you are very lucky. It is literally the [cocaine of the internet](https://www.provenmen.org/porn-damages-brain/). Imagine if your brother, or sister, or your wife, or your husband, or your favorite teacher was forced to pose without clothes for a bunch of people cause it's "just a little fun". Imagine if you had to do that. Humans were not meant for that. But because we corrupted ourselves we have spread torment and pain everywhere just for a bit of pleasure. Something warned us about that. Oh, yeah. It was the [Bible](https://www.bible.com/).
# Credits
- [Gantman/nsfw_model](https://github.com/GantMan/nsfw_model): He did what no one else had the nerve to do. I didn't want to even touch the data, so thanks for not making me have to!
- [hatoo/http-mitm-proxy](https://github.com/hatoo/http-mitm-proxy/tree/master): I modified his library for the HTTPS proxy! Couldn't have done it without this project.
- [pykeio/ort](https://github.com/pykeio/ort): Literally, most helpful project.
- My mom: Taught me how to fight lust, how to see the better way, and how to be a better man. Oh and put up with a verbally abusive man for 26 years to keep the family together. Love you!
- Jesus Christ: My savior, my redeemer, my rock, my king, my commander, and literally the sole reason I exist. Wrote the book that warned us about lust and still loves us when we ignore it. All Hail King Jesus!\
\
This was created by PonderForge, if you use this code, give credit where credit is due.\
Pslam 111:2 "Great are the works of the LORD; they are pondered by all who delight in them."
