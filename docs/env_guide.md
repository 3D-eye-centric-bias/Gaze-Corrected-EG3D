# EG3D Setup Guide

Follow these steps in your bash terminal:

1. **Change Directory to EG3D:**
   ```bash
   conda env create -f environment.yml
   ```
2. **Update Conda:**
   ```bash
   conda update -n base -c defaults conda
   conda clean --all
   ```
3. **Activate Conda Environment**
   ```bash
   conda activate eg3d
   ```
4. **Remove Existing CUDA and Install Version 11.8:**  
   **[Warning] This will require a system reboot.**
   ```bash
   sudo apt-get purge cuda* && sudo apt-get autoremove && sudo apt-get autoclean && sudo rm -rf /usr/local/cuda*
   sudo reboot
   ```
5. **Install CUDA 11.8:**
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
   sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
   sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
   sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
   sudo apt-get update
   sudo apt-get -y install cuda-11-8
   ```
6. **Install PyTorch with CUDA 11.8:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --default-timeout=150
   ```
7. **Clone L2CS-Net repository**
   ```bash
   cd eg3d/
   git clone https://github.com/Ahmednull/L2CS-Net.git
   ```
8. **Clone Custom Deep3dFaceRecon_pytorch**
   ```bash
   cd dataset_preprocessing/ffhq
   git clone https://github.com/3D-eye-centric-bias/Deep3DFaceRecon_pytorch.git
   ```
9. **Install Nvdiffrast**
    ```bash
    cd Deep3dFaceRecon_pytorch/nvdiffrast
    pip install .
    ```
10. **Install Required envireonments**  
    cd to main directory (Gaze-Corrected-EG3D)
    ```bash
    cd ../../../../
    pip install -r requirements.txt
    ```
