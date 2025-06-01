# whisperx-translator

# WhisperX & MarianMT Subtitle Generator for Unraid

## Project Description

This project provides a Dockerized Python script designed to automate the process of generating subtitles for video files and translating them into a specified target language. It leverages WhisperX for high-quality transcription (using the `large-v3` model) and Hugging Face's MarianMT for translation. The solution is optimized for use on an Unraid server, including support for GPU acceleration and recursive directory processing.

## Features

* **High-Quality Transcription:** Utilizes WhisperX with the `large-v3` model for accurate English (source language) transcription.
* **Neural Machine Translation:** Employs Hugging Face's MarianMT models for translation from English to a configurable target language (defaulting to Dutch).
* **Recursive Directory Processing:** Automatically scans a mounted volume for `.mkv` and `.mp4` video files within subdirectories.
* **Intelligent Skipping:**
    * Skips processing a video entirely if an SRT file for the **destination language** already exists and is not empty.
    * If the destination SRT is missing, it checks for an existing English (source) SRT. If found, it skips transcription and proceeds directly to translation.
    * Generates English subtitles only if they are missing or empty.
* **GPU Acceleration:** Configured to utilize NVIDIA GPUs for faster transcription and translation via CUDA.
* **Unraid Integration:** Designed for easy deployment and configuration within the Unraid Docker environment.

## Prerequisites

To successfully use this Docker solution, ensure your system meets the following requirements:

* **Unraid Server:** This solution is specifically tailored for an Unraid OS installation.
* **NVIDIA GPU:** You must have an NVIDIA graphics card compatible with CUDA.
* **NVIDIA GPU Drivers (on Unraid):** The appropriate and up-to-date NVIDIA drivers must be installed on your Unraid server. This is typically managed via an Unraid plugin or built-in functionality.
* **NVIDIA Container Toolkit (on Unraid):** This is essential for Docker to gain access to your GPU. Verify that "NVIDIA Container Toolkit" or "GPU Support" is enabled in your Unraid's Docker settings.
* **Sufficient Disk Space:**
    * **For Docker Image:** The Docker image itself (containing all installed libraries and models like WhisperX `large-v3` and MarianMT) can be several gigabytes in size. Ensure you have enough free space on the drive/cache drive where Docker stores its images.
    * **For Video Files and SRTs:** Ensure adequate free space on the share where your video files are located and where the generated SRT files will be stored.
* **Internet Connection (during initial setup and model downloads):** An active internet connection is required during the Docker image build process to download base images, Python packages, and initially, the large AI models (WhisperX and MarianMT). Subsequent runs will use cached models.
* **SSH Access or Terminal in Unraid:** You will need command-line access to your Unraid server to build the Docker image and manage the containers.

## Installation & Setup

1.  **Prepare Project Files on Unraid:**
    * Create a dedicated folder on your Unraid array, for example: `/mnt/user/appdata/whisperx-translator/`
    * Place the `Dockerfile` and the `createSrt.py` script into this folder.

2.  **Build the Docker Image:**
    * Open an SSH session to your Unraid server or use the built-in terminal in the Unraid UI.
    * Navigate to the folder where you placed your `Dockerfile` and `createSrt.py`:
        ```bash
        cd /mnt/user/appdata/whisperx-translator/
        ```
    * Execute the following command to build the Docker image. The `--no-cache` flag is recommended for the first build or if you encounter issues, to ensure all steps are run fresh.
        ```bash
        docker build --no-cache -t whisperx-translator-gpu .
        ```
    * This process may take some time, especially during the first build, as it downloads all necessary software and AI models.

## Usage

This container is designed to run as a "batch job," processing files and then exiting.

### Running from the Command Line

1.  **Navigate to your video directory:**
    Open your Unraid terminal or SSH session and change to the directory containing your video files (e.g., a TV show season folder). This is the directory that will be mounted into the container.
    ```bash
    cd "/mnt/user/TV-Series/The Big Bang Theory (2007)/Season 01/"
    ```
    (Replace with the actual path to your video files.)

2.  **Execute the Docker Run Command:**
    Once in the correct directory, run the following command. The `$(pwd)` command automatically provides the absolute path of your current directory.

    ```bash
    docker run --rm \
      --gpus all \
      -v "$(pwd)":/data \
      -e TARGET_LANGUAGE="nl" \
      whisperx-translator-gpu
    ```

    * `--rm`: Automatically removes the container once it finishes execution.
    * `--gpus all`: Grants the container access to all available NVIDIA GPUs (requires NVIDIA Container Toolkit on Unraid).
    * `-v "$(pwd)":/data`: Mounts your current host directory (`$(pwd)`) as the `/data` directory inside the container. The script will look for videos in `/data` and its subdirectories and save the SRTs next to the video files.
    * `-e TARGET_LANGUAGE="nl"`: Sets the target language for translation. You can change `"nl"` to another 2-letter language code (e.g., `"de"` for German, `"fr"` for French), provided you've adjusted the translation model in `createSrt.py` if needed (see "Customization" below).
    * `whisperx-translator-gpu`: The name of the Docker image you built.

The container will start, execute the `createSrt.py` script, and you will see progress messages directly in your terminal. The generated `.en.srt` and `.[TARGET_LANGUAGE].srt` files will appear in the same directories as your video files. The container will exit once all files are processed.

### Running via Unraid Docker GUI (Adding the Container)

1.  **Go to your Unraid Web UI** > **Docker** tab.
2.  Click **"Add Container"**.
3.  Choose **"Add an already-existing image"** or select your `whisperx-translator-gpu` image if it's in the dropdown.
4.  Configure the following parameters:
    * **Repository:** `whisperx-translator-gpu`
    * **Container Name:** `WhisperX-Subtitle-Generator` (or any name you prefer)
    * **`--gpus all` (Extra Parameters):** Under "Extra Parameters" or "Advanced View," ensure `--gpus all` is present to enable GPU access.
    * **Path/Volume Mappings:**
        * **Container Path:** `/data` (This is fixed by the script)
        * **Host Path:** Specify the absolute path to the **root directory of your video library** on Unraid (e.g., `/mnt/user/Media/Movies/` or `/mnt/user/TV-Series/`). The script will recursively search from this point.
        * **Mode:** `rw` (Read/Write)
    * **Environment Variables:**
        * Click "Add another Path, Port, Variable, Label or Device."
        * **Key:** `TARGET_LANGUAGE`
        * **Value:** Enter your desired 2-letter language code (e.g., `nl`, `de`, `fr`).
    * Click **"Apply"** to create and start the container.

### Monitoring Progress

The script prints progress messages to the standard output. You can monitor these by:
* Observing the terminal if running via SSH.
* Checking the container's logs in the Unraid Docker UI.

## Customization

### Changing the Translation Model / Target Language

The current script uses `Helsinki-NLP/opus-mt-en-nl` for English to Dutch translation. If you want to translate to a different language, you need to:

1.  **Modify `createSrt.py`:**
    * Open `createSrt.py`.
    * Change the `HF_TRANSLATE_MODEL` variable to the appropriate Hugging Face MarianMT model for your desired translation pair (e.g., `Helsinki-NLP/opus-mt-en-de` for English to German). You can find models on the [Hugging Face Hub](https://huggingface.co/models?pipeline_tag=translation&language=en&sort=downloads).
    * **Crucially, ensure the prefix in the `translate_srt` function matches the new model's expected input format.** For `Helsinki-NLP/opus-mt` models, it's typically `>>[target_lang_code]<<`. If you switch to a different model family, you might need to adjust this.

2.  **Rebuild the Docker Image:** After modifying `createSrt.py`, you *must* rebuild your Docker image using the `docker build` command (as described in "Installation & Setup") to incorporate the changes.

3.  **Set `TARGET_LANGUAGE` Environment Variable:** When running the container, set the `TARGET_LANGUAGE` environment variable to the 2-letter code of your new target language (e.g., `-e TARGET_LANGUAGE="de"`).

## License

This project is licensed under the MIT License.

---

### MIT License

Copyright (c) [Year] [Your Name or Project Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
