## Multimodal Event-IMU-Rangemeter Fusion for Robust Ego-Motion Estimation

In complex and dynamic environments, accurate and robust **ego-motion estimation** is paramount for a myriad of applications, spanning autonomous navigation, robotics, and augmented/virtual reality. Traditional frame-based cameras, while ubiquitous, often struggle in challenging scenarios characterized by high dynamic range, rapid motion, or extreme lighting conditions, frequently leading to motion blur and latency issues that compromise the fidelity of visual information.

Event cameras emerge as a compelling alternative, offering a novel paradigm for visual sensing. Unlike conventional cameras that capture intensity frames at a fixed rate, **event cameras** asynchronously report pixel-level brightness changes (events) with microsecond latency, thereby providing high temporal resolution, inherent resistance to motion blur, and exceptional high dynamic range capabilities. Despite these advantages, event data is sparse and irregular, posing unique challenges for feature extraction and state estimation. Moreover, no single sensor modality is universally robust; **inertial measurement units (IMUs)** offer direct measurements of angular velocity and linear acceleration, providing short-term motion cues robust to visual ambiguities, while **rangemeters** furnish absolute distance measurements, crucial for scale and altitude estimation.

This work addresses the critical problem of real-time, robust 6-Degree-of-Freedom (6-DoF) ego-motion estimation in challenging environments by effectively fusing information from event cameras, IMUs, and rangemeters. We propose an **end-to-end deep learning framework** designed to leverage the complementary strengths of these disparate sensor modalities. Our approach integrates a sophisticated 3D Convolutional Neural Network (CNN) for spatio-temporal event feature learning, recurrent neural networks for sequential IMU and rangemeter data processing, and a novel **cross-modal attention mechanism** for dynamic sensor fusion.

The proposed system takes as input:
1.  An event stream $\mathcal{E} = \{ (x_k, y_k, t_k, p_k) \}_{k=1}^{N_e}$, where $(x_k, y_k)$ are pixel coordinates, $t_k$ is the timestamp, and $p_k \in \{-1, +1\}$ denotes the event polarity (brightness increase or decrease).
2.  A sequence of IMU readings $\mathbf{I}_{\text{seq}} = \{ [a_x, a_y, a_z, \omega_x, \omega_y, \omega_z]^T \}_j$, comprising linear accelerations and angular velocities.
3.  A sequence of rangemeter readings $\mathbf{R}_{\text{seq}} = \{ r_j \}_j$, representing distance measurements.

The objective is to accurately predict the 6-DoF ego-motion state $\hat{\mathbf{x}}_t = [\hat{p}_x, \hat{p}_y, \hat{p}_z, \hat{v}_x, \hat{v}_y, \hat{v}_z]^T$ (3D position and 3D linear velocity) at time $t$.

---

## Methodology

Our architecture, which we term the **Multimodal Event-IMU-Rangemeter Network (EMMNet)**, comprises three dedicated encoders for each modality, followed by a fusion module and a regression head.

### 1. Event Encoder

To process the asynchronous event stream, raw events are first aggregated into a dense spatio-temporal tensor. For each prediction timestamp $t$, events within a fixed preceding time window $\Delta t_{\text{event}}$ are collected and discretized into $T$ temporal bins. For each bin $j \in \{1, \dots, T\}$ and pixel $(u, v)$, events of positive ($p_k=+1$) and negative ($p_k=-1$) polarities are counted:
$$E_{j,u,v,+1} = \sum_{k: t_k \in \text{bin } j, p_k=+1} 1$$
$$E_{j,u,v,-1} = \sum_{k: t_k \in \text{bin } j, p_k=-1} 1$$This yields an input event tensor $\mathbf{E} \in \mathbb{R}^{2 \times T \times H \times W}$, where $H$ and $W$ are the spatial dimensions. This tensor is then fed into a **3D Convolutional Neural Network**, inspired by ResNet architectures, which we refer to as the **EventEncoder**. It consists of initial `Conv3d` layers followed by `BatchNorm3d`, `ReLU` activations, and `MaxPool3d` layers for hierarchical feature extraction. Subsequent `ResNet3DBlock`s deepen the network, each block implementing a residual connection:$$\mathbf{y} = \text{ReLU}(\text{BN}(\text{Conv3D}_2(\text{ReLU}(\text{BN}(\text{Conv3D}_1(\mathbf{x})))))) + \text{Shortcut}(\mathbf{x})$$
where $\text{Shortcut}(\mathbf{x})$ is either an identity mapping or a `Conv3d` layer for dimensionality matching. This encoder extracts a compact event feature vector $\mathbf{f}_E \in \mathbb{R}^{D_E}$.

### 2. IMU Encoder

The IMU data, typically sampled at a high frequency, is processed as a sequence of readings over a window $\Delta t_{\text{imu}}$. A **Long Short-Term Memory (LSTM)** network is employed to capture temporal dependencies within this sequence, mapping the IMU readings $\mathbf{I}_{\text{seq}}$ to a feature vector $\mathbf{f}_I \in \mathbb{R}^{D_I}$:
$$\mathbf{f}_I = \text{LSTM}(\mathbf{I}_{\text{seq}})$$

### 3. Rangemeter Encoder

Similarly, rangemeter readings are fed into another **LSTM** network to extract a rangemeter-specific feature vector $\mathbf{f}_R \in \mathbb{R}^{D_R}$:
$$\mathbf{f}_R = \text{LSTM}(\mathbf{R}_{\text{seq}})$$

### 4. Fusion Module (Cross-Modal Attention)

The distinct feature representations from the EventEncoder, IMUEncoder, and RangemeterEncoder ($\mathbf{f}_E, \mathbf{f}_I, \mathbf{f}_R$) are then combined using a **Cross-Modal Attention mechanism**. This module dynamically assesses the relevance of each modality's features, enabling the network to weigh sensor contributions based on the current context. A **Multihead Attention (MHA)** block is utilized, where each feature vector $\mathbf{f}_m$ (for $m \in \{E, I, R\}$) is linearly projected into query, key, and value representations:
$$\mathbf{Q}_m = \mathbf{f}_m \mathbf{W}_Q^{(m)}, \quad \mathbf{K}_m = \mathbf{f}_m \mathbf{W}_K^{(m)}, \quad \mathbf{V}_m = \mathbf{f}_m \mathbf{W}_V^{(m)}$$
The attention mechanism then computes weighted sums of values based on similarity scores between queries and keys, facilitating information exchange across modalities. The output of the attention module is a fused feature vector $\mathbf{f}_{\text{fused}}$.

### 5. Regression Head

Finally, the concatenated or fused feature vector $\mathbf{f}_{\text{fused}}$ is passed through a **Multi-Layer Perceptron (MLP)** with ReLU activations and Dropout layers to predict the 6-DoF ego-motion state $\hat{\mathbf{x}}_t$:
$$\hat{\mathbf{x}}_t = \text{MLP}(\mathbf{f}_{\text{fused}})$$The entire network is trained end-to-end by minimizing the **Mean Squared Error (MSE)** between the predicted state $\hat{\mathbf{x}}_t$ and the ground truth state $\mathbf{x}_t$:$$\mathcal{L} = \left\| \hat{\mathbf{x}}_t - \mathbf{x}_t \right\|_2^2$$

---

## Contributions

This work presents a robust framework for real-time ego-motion estimation that uniquely benefits from: (1) an **end-to-end deep learning approach** capable of learning directly from raw event streams, IMU, and rangemeter data; (2) the utilization of **3D CNNs to effectively capture spatio-temporal patterns** from event data; and (3) an **adaptive cross-modal attention mechanism** that intelligently fuses information from complementary sensors. Our experimental evaluations demonstrate the efficacy of this multimodal fusion strategy in providing accurate and resilient pose and velocity estimates, particularly valuable for demanding application scenarios where single-modality solutions fall short.