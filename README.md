
# CARE-GS

**Camera-Aligned Refinement and Enhancement for Gaussian Splatting**

CARE-GS is the monocular static 3D Gaussian initialization and refinement module of the MERCY project.

CARE-GS first uses **Hunyuan3D 3.1** to generate an initial 3D mesh from a single RGB image. The mesh is then converted into an initial 3D Gaussian Splatting representation (G0) using **Mesh2Splat**.

Because the reconstructed 3D model may not perfectly match the original input image, CARE-GS performs fixed-intrinsics virtual camera alignment, RGB-guided Gaussian refinement, and geometric continuity enhancement to improve the consistency between the Gaussian representation and the reference image.

The resulting corrected Gaussian model (G0*) serves as the static initialization for subsequent dynamic 4D Gaussian reconstruction in MERCY.

## 1. Project Structure

```text
CARE-GS/
├── Stage1/
│   ├── train_stage1.py
│   ├── camera_alignment.py
│   ├── gaussian_refinement.py
│   ├── gaussian_utils/
│   └── *_stage1.ini
│
├── Stage1_Visualization_UI/
│   ├── app.py
│   └── render_engine.py
│
├── scene/
├── gaussian_renderer/
├── utils/
├── arguments/
├── submodules/
│
├── third_party/
│   ├── Hunyuan3D-2.1/
│   └── mesh2splat/
│
├── requirements.txt
└── README.md
```

## 2. Installation

### 2.1 Clone the Repository

Clone CARE-GS together with its Git submodules:

```bash
git clone --recurse-submodules https://github.com/Kiki0719/GRACE-GS.git CARE-GS
cd CARE-GS
```

If the repository has already been cloned:

```bash
git submodule update --init --recursive
```

Third-party repositories:

- [Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)
- [Mesh2Splat](https://github.com/electronicarts/mesh2splat)

**Note:** The clone URL above uses the original GRACE-GS repository address. Replace it with the CARE-GS repository URL after renaming the GitHub repository.

### 2.2 Create the Python Environment

Recommended development environment:

- Linux
- Python 3.10
- PyTorch 2.1.2
- CUDA 12.1
- NVIDIA GPU

Create a Conda environment:

```bash
conda create -n care-gs python=3.10 -y
conda activate care-gs
```

It is recommended to maintain separate Python environments for Hunyuan3D and CARE-GS to avoid dependency conflicts.

### 2.3 Install Python Dependencies

Install a compatible PyTorch version for your CUDA environment.

Then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Some dependencies may require adjustments depending on the operating system, CUDA toolkit, and compiler versions.

### 2.4 Install CUDA Extensions

CARE-GS uses differentiable Gaussian rasterization and CUDA-accelerated nearest-neighbor computation.

Install the required extensions:

```bash
python -m pip install ./submodules/simple-knn

python -m pip install ./submodules/depth-diff-gaussian-rasterization
```

Verify installation:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

python -c "import diff_gaussian_rasterization; import simple_knn; print('CUDA extensions OK')"
```

## 3. Static Gaussian Initialization

Before CARE-GS refinement, the initial Gaussian representation G0 must be generated using Hunyuan3D and Mesh2Splat.

The initialization pipeline is:

```text
Single RGB Image
       |
       v
  Hunyuan3D 2.1
  Shape + Texture
       |
       v
   Initial Mesh
       |
       v
    Mesh2Splat
       |
       v
 Initial Gaussian G0
       |
       v
     CARE-GS
```

### 3.1 Hunyuan3D Environment Preparation

Enter the Hunyuan3D directory:

```bash
cd third_party/Hunyuan3D-2.1
```

Follow the official installation instructions:

https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1

Install its dependencies and download the required model checkpoints.

Hunyuan3D 2.1 provides single-image 3D shape generation and texture generation.

### 3.2 Generate the Initial Mesh

Input a reference RGB image into Hunyuan3D to generate an initial textured mesh.

Recommended output structure:

```text
datasets/
└── OBJECT_NAME/
    ├── OBJECT_NAME.png
    └── OBJECT_NAME.glb
```

Where:

- `OBJECT_NAME.png`: Reference RGB image.
- `OBJECT_NAME.glb`: Generated 3D mesh.

The mesh contains the initial geometry and appearance information inferred from the reference image.

### 3.3 Mesh2Splat Environment Preparation

Mesh2Splat converts a textured mesh into a 3D Gaussian Splatting representation.

Official repository:

https://github.com/electronicarts/mesh2splat

Enter the Mesh2Splat directory:

```bash
cd third_party/mesh2splat
```

For Linux, install the required development packages:

```bash
sudo apt update

sudo apt install -y \
  build-essential \
  cmake \
  pkg-config \
  git \
  libfreeimage-dev \
  libglew-dev \
  libglfw3-dev \
  libgl1-mesa-dev \
  libxinerama-dev \
  libxcursor-dev \
  libxi-dev \
  libxxf86vm-dev
```

Build Mesh2Splat:

```bash
mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release

cmake --build . -j$(nproc)
```

Mesh2Splat requires a compatible graphics environment. Headless Linux servers may require additional OpenGL or display configuration.

Windows users should refer to the official Mesh2Splat build documentation.

### 3.4 Convert Mesh to Gaussian G0

Use Mesh2Splat to convert the Hunyuan3D mesh into a Gaussian representation.

```text
Hunyuan3D Output
       |
       v
OBJECT_NAME.glb
       |
       v
   Mesh2Splat
       |
       v
OBJECT_NAME.ply
       |
       v
Initial Gaussian G0
```

The output should be a valid 3D Gaussian Splatting PLY file rather than a conventional mesh PLY.

Before CARE-GS training, verify the compatibility of the Gaussian representation, including:

- Gaussian positions
- Scaling parameters
- Rotation parameters
- Opacity
- Spherical harmonics coefficients
- Coordinate conventions

If the generated PLY format differs from the GaussianModel representation used by CARE-GS, format conversion is required.

### 3.5 Prepare Input Data

Organize the generated files as follows:

```text
datasets/
└── OBJECT_NAME/
    ├── OBJECT_NAME.png
    ├── OBJECT_NAME.glb
    ├── OBJECT_NAME.ply
    └── camera_intrinsics.json
```

Input descriptions:

| File | Description |
|---|---|
| `OBJECT_NAME.png` | Reference RGB image |
| `OBJECT_NAME.glb` | Initial 3D mesh generated by Hunyuan3D |
| `OBJECT_NAME.ply` | Initial Gaussian G0 generated by Mesh2Splat |
| `camera_intrinsics.json` | Fixed camera intrinsic parameters |

CARE-GS Stage1 mainly requires the reference RGB image, the Gaussian G0, and the camera intrinsics.

The mesh is retained as an intermediate reconstruction artifact.

## 4. CARE-GS Pipeline

CARE-GS consists of two major optimization stages:

1. Fixed-Camera Alignment
2. Gaussian Refinement

### 4.1 Fixed-Camera Alignment

The initial Gaussian model generated from a monocular image may not be aligned with the original camera viewpoint.

CARE-GS therefore estimates a virtual camera pose while keeping the camera intrinsic parameters fixed.

Given an initial Gaussian representation G0 and the original RGB image, camera alignment optimizes the virtual camera extrinsics to improve the agreement between rendered and observed appearances.

The optimization considers image similarity and foreground silhouette consistency.

Output:

```text
fixed_camera.json
```

This file contains the estimated virtual camera parameters for subsequent Gaussian refinement and rendering.

### 4.2 Gaussian Refinement

Even after camera alignment, the initial Gaussian model may contain:

- Inaccurate foreground silhouettes
- Missing visible surface regions
- Local geometric inconsistencies
- Appearance differences from the original image
- Discontinuous or insufficiently covered Gaussian surfaces

CARE-GS refines the Gaussian representation using RGB supervision, silhouette constraints, and geometric regularization.

The optimization aims to improve reference-view reconstruction while preserving plausible 3D structure.

Output:

```text
corrected_g0.ply
```

The refined Gaussian representation is denoted G0*.

### 4.3 Complete Pipeline

```text
               Single RGB Image
                      |
                      v
                Hunyuan3D 2.1
                Shape + Texture
                      |
                      v
                 Initial Mesh
                      |
                      v
                  Mesh2Splat
                      |
                      v
              Initial Gaussian G0
                      |
                      v
                   CARE-GS
                      |
          +-----------+-----------+
          |                       |
          v                       |
   Fixed-Camera Alignment         |
   (Fixed Intrinsics)             |
          |                       |
          v                       |
   Optimized Camera               |
          |                       |
          +-----------+-----------+
                      |
                      v
              Gaussian Refinement
         RGB / Silhouette / Geometry
                      |
                      v
              Corrected Gaussian G0*
                      |
                      v
              MERCY Dynamic 4DGS
```

## 5. Configuration

Return to the CARE-GS root directory:

```bash
cd /path/to/CARE-GS

conda activate care-gs
```

Copy an existing configuration file:

```bash
cp Stage1/fan_stage1.ini Stage1/custom_stage1.ini
```

Modify the configuration according to your input data and output directories.

### 5.1 Input and Output Paths

```ini
[Path]

source_path = /path/to/datasets/OBJECT_NAME

initial_gaussian_ply = /path/to/datasets/OBJECT_NAME/OBJECT_NAME.ply

t0_image_path = /path/to/datasets/OBJECT_NAME/OBJECT_NAME.png

t0_mask_path = None

camera_file = /path/to/outputs/OBJECT_NAME/round1/alignment/fixed_camera.json

model_path = /path/to/outputs/OBJECT_NAME/round1/refinement

diagnostics_path = /path/to/outputs/OBJECT_NAME/round1/refinement/diagnostics
```

### 5.2 Camera Alignment Configuration

```ini
[Camera Alignment]

camera_intrinsics_path = /path/to/datasets/OBJECT_NAME/camera_intrinsics.json

camera_alignment_output = /path/to/outputs/OBJECT_NAME/round1/alignment

camera_prior_path = None
```

### 5.3 Stage1 Pipeline Configuration

```ini
[Stage1 Pipeline]

stage1_pipeline_root = /path/to/outputs

stage1_round1_dir = /path/to/outputs/OBJECT_NAME/round1

stage1_refinement_output = /path/to/outputs/OBJECT_NAME/round1/refinement

stage1_final_dir = /path/to/outputs/OBJECT_NAME/final
```

The examples above show the primary path settings.

Keep the remaining training and optimization parameters from the original INI configuration.

Replace all `/path/to/...` placeholders with valid absolute paths.

## 6. Training

After generating the initial Gaussian G0 and preparing the configuration file, run CARE-GS:

```bash
conda activate care-gs

cd /path/to/CARE-GS

CUDA_VISIBLE_DEVICES=0 python -u Stage1/train_stage1.py \
  --config Stage1/custom_stage1.ini
```

Training sequentially executes:

1. Camera Alignment
2. Gaussian Refinement

### 6.1 Output Structure

A typical output directory is:

```text
outputs/
└── OBJECT_NAME/
    └── round1/
        ├── alignment/
        │   ├── fixed_camera.json
        │   └── diagnostic_views/
        │
        └── refinement/
            ├── corrected_g0.ply
            └── diagnostics/
```

Main outputs:

| Output | Description |
|---|---|
| `fixed_camera.json` | Optimized virtual camera parameters |
| `corrected_g0.ply` | Refined Gaussian representation G0* |
| `diagnostics/` | Training diagnostics and visualization results |

## 7. Visualization UI

CARE-GS provides a Streamlit-based interface for inspecting Gaussian reconstruction and refinement results.

### 7.1 Launch the UI

```bash
conda activate care-gs

cd /path/to/CARE-GS

streamlit run Stage1_Visualization_UI/app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

Local access:

```text
http://localhost:8501
```

For remote servers, access the corresponding mapped port.

### 7.2 Auto Detail Comparison

The Auto Detail Comparison page provides:

- Original RGB / Original G0 / Corrected G0 comparison
- Automatic selection of four local detail regions
- 5×3 scientific comparison figures
- PSNR, SSIM, MAE, and Mask IoU evaluation
- Exportable figures and metric tables

### 7.3 Four-View Scientific Figure

The multi-view visualization page supports:

- Front view
- Left view
- Right view
- Back view

It provides visual comparisons between the initial Gaussian G0 and the corrected Gaussian G0*.

### 7.4 UI Data Configuration

The visualization interface supports manually specifying:

- Original Gaussian PLY
- Corrected Gaussian PLY
- Fixed Camera JSON
- Stage1 INI
- Original RGB image
- Optional target mask

For preset-based dataset discovery, edit:

```text
Stage1_Visualization_UI/app.py
```

Set the corresponding directories:

```python
preset_root = Path("/path/to/outputs")

data_root = Path("/path/to/datasets")
```

Restart Streamlit after changing the paths.

## 8. Main Source Files

| File | Description |
|---|---|
| `Stage1/train_stage1.py` | Main Stage1 training pipeline |
| `Stage1/camera_alignment.py` | Fixed-intrinsics camera alignment |
| `Stage1/gaussian_refinement.py` | RGB-guided Gaussian refinement |
| `Stage1/gaussian_utils/` | Gaussian densification and geometry processing |
| `Stage1_Visualization_UI/app.py` | Streamlit visualization interface |
| `Stage1_Visualization_UI/render_engine.py` | Gaussian loading and rendering |
| `scene/` | Gaussian model and scene management |
| `gaussian_renderer/` | Differentiable Gaussian renderer |
| `utils/` | General utility functions |
| `submodules/` | CUDA extensions |
| `third_party/Hunyuan3D-2.1/` | Single-image 3D generation |
| `third_party/mesh2splat/` | Mesh-to-Gaussian conversion |

## 9. Important Notes

1. Hunyuan3D, Mesh2Splat, and CARE-GS may require separate environments and dependency installation.

2. Download the required Hunyuan3D checkpoints according to the official documentation.

3. Mesh2Splat requires a compatible C++ compiler, CMake, and graphics environment.

4. CARE-GS expects a Gaussian Splatting PLY representation, not a conventional triangle mesh PLY.

5. Ensure that all input and output paths in the INI configuration are valid.

6. Camera alignment optimizes the virtual camera pose while keeping the intrinsic parameters fixed.

7. Static Gaussian refinement primarily uses RGB, silhouette, and geometry constraints.

8. Event streams are not mandatory inputs for CARE-GS Stage1.

9. Gaussian initialization quality and reference-view alignment are important for subsequent motion estimation.

10. Verify the compatibility of the Gaussian representation before training.

## 10. MERCY Project

CARE-GS serves as the static Gaussian initialization and refinement module of MERCY.

**MERCY — Monocular Event-based Locally Rigid Motion-Coherent 4D Gaussian Reconstruction**

MERCY combines static 3D Gaussian initialization with monocular motion information and event-based observations to support dynamic 4D Gaussian reconstruction.

CARE-GS provides the initial corrected Gaussian representation G0* used by the subsequent reconstruction stages.

MERCY GitHub:

https://github.com/SuleynanAuir/MERCY-Monocular-Event-Locally-RigidMotion-Coherent-4DGS-Reconstruct

## Acknowledgements

CARE-GS uses the following open-source projects for its static initialization pipeline.

### Hunyuan3D 2.1

Tencent Hunyuan Team

https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1

### Mesh2Splat

Electronic Arts SEED

https://github.com/electronicarts/mesh2splat

Both repositories are referenced through Git submodules.

The use and redistribution of third-party source code, pretrained models, and associated resources are subject to their respective licenses and usage terms.
