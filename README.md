[<img src="https://raw.githubusercontent.com/PonderForge/LustBlock/main/Logo.png" width="400">](https://raw.githubusercontent.com/PonderForge/LustBlock/main/Logo.png)\
High Speed AI Porn and Sexy Image Proxy Filter directly on your Computer! 
![demo](https://raw.githubusercontent.com/PonderForge/LustBlock/main/demo.gif)\
(We are not associated with clideo.com, we just don't have a good video editor)
# Install
## Ubuntu - Noble or Newer
1. Go to our latest release and download it.\
`wget https://github.com/PonderForge/LustBlock/releases/download/v0.1.0/ubuntu-noble-x64.deb -o lustblock.deb`
2. Unzip/Extract it using your favorite extractor\
`sudo dpkg -i lustblock.deb`
3. Run the main program\
You can run it by typing: `lustblock`
## Linux - All other distros
1. Go to our latest release and download it.\
`wget https://github.com/PonderForge/LustBlock/releases/download/v0.1.0/linux-all-x64.tar.gz -o lustblock.tar.gz`
2. Unzip/Extract it using your favorite extractor
`tar -xvzf lustblock.tar.gz`
3. Install OpenCV lib via system package manager
4. Change into the release directory and run the main program
You can run it using `cd lustblock && ./lustblock`
## Windows
1. Download windows-x64-setup.exe
2. Follow instructions through the prompts
3. Clear Your Browser Cache
4. Skip the Post install below, if you are only installing for your system
## Post Install
1. Configure your operating system to use the proxy, via the server's local or public IP, as set per the config.yaml:
- On Ubuntu, follow [this tutorial](https://phoenixnap.com/kb/ubuntu-proxy-settings)
- On Windows, follow [this tutorial](https://support.microsoft.com/en-us/windows/use-a-proxy-server-in-windows-03096c53-0554-4ffe-b6ab-8b1deee8dae1)
- On Android, follow [this tutorial](https://proxyway.com/guides/android-proxy-settings)
- On iOS, follow [this tutorial](https://libertyshield.kayako.com/article/32-manual-proxy-ios-iphone-and-ipad)
2. Finally, to use HTTPS and to prevent your browser from going into lockdown, install pub.crt, a fake cert, which is in your OS's config directory, which is either `C:/Users/{user}/AppData/Roaming/LustBlock` or `/home/{user}/.config/LustBlock`:
- On Ubuntu, follow [this tutorial](https://askubuntu.com/questions/73287/how-do-i-install-a-root-certificate/94861#94861)
- On Windows, follow [this tutorial](https://web.archive.org/web/20160612045445/http://windows.microsoft.com/en-ca/windows/import-export-certificates-private-keys#1TC=windows-7)
- On Android, follow [this tutorial](http://wiki.cacert.org/FAQ/ImportRootCert#Android_Phones_.26_Tablets)
- On iOS, follow [this tutorial](http://jasdev.me/intercepting-ios-traffic)
# Compiling
If you want to compile with more of ONNX's Execution providers such as DirectX, or you just want to mess with the code, follow this tutorial.
1. Download the source code either via GitHub desktop, or by running:\
`git clone https://github.com/PonderForge/LustBlock.git && cd LustBlock`
2. Write any edits to the code as you wish
3. Install Clang, OpenCV and NASM (if you're on Windows, good luck.)
4. Next use Cargo to build via:\
`cargo build`
5. Go to the target/debug directory and run LustBlock
# Configuration 
We have 2 methods of configuration, either via config.yaml which exposes basic config, or via recompilation.
## Execution Providers
LustBlock uses the ORT, an ONNX library for Rust, which provides many different execution providers, or modules that make the AI Proxy run faster. CUDA and TensorRT is included by default in releases and configurable via config.yaml, the rest of the execution providers you must compile yourself.
You can add one by:
1. Finding the Execution Provider and if ORT supports it, [here](https://ort.pyke.io/perf/execution-providers).
2. If ORT does support it, you have to add the Cargo feature in the Cargo.toml
3. Then include it in the ort "use" statement in src/main.rs:\
`use ort::{inputs, {put the provider name here}, CPUExecutionProvider, CUDAExecutionProvider, GraphOptimizationLevel, Session, SessionOutputs, TensorRTExecutionProvider};`\
More on the list of Execution provider structs [here](https://docs.rs/ort/2.0.0-rc.2/ort/index.html?search=ExecutionProvider)
4. Finally add it to the execproviders array in src/main.rs like so:\
`let exec_providers = [ {put the provider name here}::default().build(),`
5. You may need to install the required libraries for the execution provider, like with CUDA you need CUDA 12 and cuDNN 9, more on this [here](https://ort.pyke.io/perf/execution-providers).
## Thresholds
So AIs are not completely accurate (ours is 93%), and LustBlock only returns the likelihood of the image being a certain class of image. This means that an image is identified as NSFW if the likelihood of the image being Hentai, Porn, or Sexy is a number higher than the numbers as set in config.yaml. You can change these to your preference as long as you know that an increase in thresholds could increase how many NSFW images slip through, and a decrease in thresholds could increase neutral images getting removed.
## Per-Site Settings
Since we run 2 different AIs on every image, we thought it best to add per-site settings!\
In config.yaml, at the bottom, you can add your site domains and set whether or not the human-based filter runs, the image-based filter runs, or whether the site is allowed in the first place! (Blacklist) Personally, I suggest blacklisting all known porn sites on there, we'll add in our own global blacklist at some point.
# TODO
1. Better Human and Image Replacement (My Cat is great but he's stretched too thin)
2. Improve Human OBB with OpenImages Dataset
3. Add Human Segmentation to give people clothes where they need it (Cause they're too lazy to do it themselves?)
4. Video Scanning: It'll be slow as molasses but hey, sexual free content!
5. Text Scanning: Using FastText and OCR
6. Encrypted Porn Website blocker: Since many websites are 100% porn, we want to add a encrypted list of websites that should be completely blocked and blacklisted.
7. Client and Server based Proxy: This will allow phones and older computers to run the proxy by configuring the device and then connecting to a remote computer that runs the server.
8. Scan POST requests: Cause people also send out images, we need to scan those for sexting, and other forms of porn and sexual images.
9. Better Client Integration with OS: Having a program visibally run is annoying, we need to hide it in the system tray, and have setting editing via GUI. And anti-virus exceptions!
# Contributing
I know that there's a lot of bugs, but that's were you come in! I need beta testers, programmers, hackers, etcetera, to find problems. Beta testers, please don't purposely find sexy images, but if you happen to come across them with LustBlock on, submit a issue (just not the problematic photo)! Programmers, I'm just one man, plus my calculus professor keeps giving me homework, so please, if you have optimizations, submit a issue! Hackers, I know the program is insecure, so make it secure! I'm not a expert, but I am willing to work with people! THX!
If you want to donate, donate to my church at [firste.org](https://firste.org).
# Why?
Since you are this far into the README, you must be interested in the motivation for this project.
Lust by definition is "very strong sexual desire" for another person as almost everyone experiences. Left unchecked, it starts to destroy you internally which will begin to affect others around you. Porn and Sexual images are one of the easiest things to lust after because it is cheap, hideable, and spreads faster than my dog can run. Even a simple Google search could bring up results that could spiral you down a path of torment and destruction, just due to the girl or boy that was taught that they are nothing more than meat. That is why LustBlock was created. Thanks to the help of GantMan's NSFW models, we can quickly and accurately detect and remove NSFW images from the web before it reaches you and your loved ones. Did you know that when one starts to see porn for "fullfillment", their chance of getting a divorce rate [doubles](https://www.science.org/content/article/divorce-rates-double-when-people-start-watching-porn)? If you have not had to fight lust, you are very lucky. It is literally the [cocaine of the internet](https://www.provenmen.org/porn-damages-brain/). Imagine if your brother, or sister, or your wife, or your husband, or your favorite teacher was forced to pose, naked, for a bunch of people cause it's "just a little fun". Imagine if you had to do that. Humans were not meant for that. But because we corrupted ourselves we have spread torment and pain everywhere just for a bit of pleasure. Something warned us about that. Oh, yeah. It was the [Bible](https://www.bible.com/) and literally thousands of years of history.
# Credits
- [hatoo/http-mitm-proxy](https://github.com/hatoo/http-mitm-proxy/tree/master): I modified his library for the HTTPS proxy! Couldn't have done it without this project.
- [pykeio/ort](https://github.com/pykeio/ort): Literally, most helpful project for highspeed AI inferencing.
- [AdamCodd/vit-base-nsfw-detector](https://huggingface.co/AdamCodd/vit-base-nsfw-detector): The NSFW Classifier AI behind it all, couldn't have done it without you!
- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics): Used the YOLOV11 architecture for human boxing for per-human porn detection and removal.
- My mom: Taught me how to fight lust, how to see the better way, and how to be a better man. Oh and put up with a verbally abusive man for 26 years to keep the family together. Love you!
- Jesus Christ: My savior, my redeemer, my rock, my king, my commander, and literally the sole reason I exist. Wrote the book that warned us about lust and still loves us when we ignore it. All Hail King Jesus!\
\
This was created by PonderForge, if you use this code, give credit where credit is due.\
Pslam 111:2 "Great are the works of the LORD; they are pondered by all who delight in them."
