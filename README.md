# One-Way Communication System with Low Error Rate and Low Latency

## Overview

This project implements a one-way communication system using the UDP protocol, designed for low error rates and low latency. It utilizes Reed-Solomon forward error correction (FEC) optimized for erasure channels, and applies Fermat Number Transform (FNT) to achieve efficient encoding and decoding with complexity O(NlogN). The system is accelerated using parallel computing on modern hardware:

- ISPC + AVX2 (Intel CPU)
- CUDA (Nvidia GPU)

### Real-World Applications

- Secure one-way data transmission from SCADA systems to office networks (e.g., EVN)
- Low-latency multimedia communication (e.g., real-time video, online gaming)

## Author

- Name: Nguyễn Vũ Khánh Uy  
- Student ID: 102210240  
- Class: 21TCLC_DT3  
- Advisor: MSc. Nguyễn Thế Xuân Ly  
- University: University of Science and Technology – University of Danang

## Objectives

- Build a one-way communication system with error correction and minimal latency
- Evaluate the parallel processing capability of modern CPUs and GPUs

## Technologies Used

- Languages: C/C++, Java
- CPU Target: Intel Core i5-12400 (AVX2 i32x8)
- GPU Target: Nvidia RTX 3060 (Ampere)
- Frameworks and Tools:
  - ISPC – Intel SPMD Program Compiler
  - CUDA – Nvidia GPU programming
  - Intel oneAPI Threading Building Blocks (TBB)
  - GNS3 – Network simulation
  - Nsight Compute and Nsight Systems – Profiling and optimization

## Error Correction

- Type: Reed-Solomon (Erasure Code)
- Field: GF(65537)
- Performance:
  - Encoding/Decoding Complexity: O(NlogN)
  - High resilience to burst loss and erasures
- Reliability: Guaranteed packet loss < 10⁻⁶ when sending 1 TB of data in a 20% packet loss environment

## System Versions

### CPU Version
- Targeted for low-latency, small-packet business applications
- Encoding speed: 405 MB/s  
- Decoding speed: 22 MB/s

### GPU Version
- Targeted for high-throughput data transmission
- Encoding speed: 1.1 GB/s  
- Decoding speed: 61 MB/s

## System Workflow

1. Encoding: Split input data into blocks, encode using Reed-Solomon and FNT
2. Transmission: Send data via UDP over a one-way physical link
3. Decoding: Reconstruct original data using error correction from received packets

## Testing and Evaluation

- Simulated with GNS3 under:
  - 20% packet loss
  - 5% packet corruption
- Benchmarked with file sizes: 500 KB to 10 MB
- Tools: iperf3, Nsight, custom console application
- Metrics evaluated:
  - Throughput
  - Latency
  - Cache hit rate
  - GPU warp activity

## Results

- Stable system operation under harsh network conditions
- Efficient parallelization with ISPC on CPU and CUDA on GPU
- High data integrity and low retransmission without using TCP

## Conclusion

This project shows the viability of building a high-performance, low-latency, one-way communication system using modern error correction techniques and parallel hardware. It is well-suited for secure industrial data pipelines and real-time transmission.
