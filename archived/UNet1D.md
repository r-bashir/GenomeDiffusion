        Input SNP Sequence (1D)
                    │
        ┌───────────▼───────────┐
        │   Initial ConvBlock   │  (Expands feature representation)
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │  Downsampling Path    │  (Extracts hierarchical features)
        │ ┌───────────────────┐ │
        │ │ ResNet Block      │ │
        │ │ ResNet Block      │ │
        │ │ DownsampleConv    │ │
        │ └───────────────────┘ │
        │ ┌───────────────────┐ │
        │ │ ResNet Block      │ │
        │ │ ResNet Block      │ │
        │ │ DownsampleConv    │ │
        │ └───────────────────┘ │
        │   ... More Layers ... │
        └───────────┬───────────┘
                    │
          ┌─────────▼─────────┐
          │   Bottleneck      │    (Captures deep representations)
          │  ResNet Block     │
          │  ResNet Block     │
          └─────────┬─────────┘
                    │
        ┌───────────▼───────────┐
        │  Upsampling Path      │  (Reconstructs the sequence)
        │ ┌───────────────────┐ │
        │ │ UpsampleConv      │ │
        │ │ ResNet Block      │ │
        │ │ ResNet Block      │ │
        │ └───────────────────┘ │
        │ ┌───────────────────┐ │
        │ │ UpsampleConv      │ │
        │ │ ResNet Block      │ │
        │ │ ResNet Block      │ │
        │ └───────────────────┘ │
        │   ... More Layers ... │
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │  Final Conv1D Layer   │  (Maps to output space)
        └───────────┬───────────┘
                    │
        Output SNP Sequence (1D)


https://chatgpt.com/share/67c6bb2d-8288-800d-a461-52eca39c7d6a