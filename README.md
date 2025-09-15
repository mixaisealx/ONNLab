# Obfuscated Neural Network Lab

This framework was developed for conducting experiments. It offers flexible configuration of neural network architectures, integration of custom components, and complete control over training and inference processes. Framework was designed conceptually as an analogue of optical laboratory benches, emphasizing systematic flexibility to support the design of any architecture and training method.
> Drawing an analogy with techniques used to protect the business logic of program code, the proposed approach to safeguarding the internal knowledge of neural networks (that govern their decision-making logic) is called "obfuscation" in this project.

The framework largely consists of header files (with the `.h` extension); however, some functionality resides in compilable source files (with the `.cpp` extension), whose names match the corresponding header files. For any experiment to work properly, it is recommended to compile all framework source files to avoid linking issues. For convenience, it is recommended to add the prefix `exp_` to the names of experiment source files. It is also recommended to place the experiment entry point declaration in the `onnlab/onnlab.h` file.

<details>
  <summary>Non-essential Info</summary>
  
  *In a sense, the concept of the project is inspired by the setup of most optics laboratories (specifically, optical tables) - providing maximum flexibility for a wide variety of experiments.*
</details>

**Outline:**

1. [Run Guide](#run-guide)
2. [Experimental Data Processing](#experimental-data-processing)
3. [Framework Documentation](#framework-documentation)

## Run Guide

The framework is a **C++20** project, you can build it using a **Makefile** or **Visual Studio**. 

Before the run, you should ensure the **MNIST** dataset have the following paths: `mnist-in-csv/mnist_train.csv` and `mnist-in-csv/mnist_test.csv` (it is assumed that you will download a dataset and place it's files here).

#### Building using Visual Studio

1. You need **Microsoft Visual Studio** setup with C++ development components on your system (the developer had Visual Studio 2022).

2. You need to open the project file: `onnlab/onnlab.vcxproj`.

3. You need to select the `x86` configuration (only this configuration has been tested, and so far there has been no need for x64) and the `Release` build type (to apply optimizations).

4. Next, you can run the compiled file. Note that the experimental code assumes that the executable file will be located at path `xxx/yyy` (for example, `Release/onnlab.exe`).

#### Building using Visual Studio

1. Make sure you have the following tools installed:
    - `g++` (version 10 or newer)
    - `make`
    - `g++-multilib`
    
    If not, on Ubuntu/Debian you may run the following command to install: `sudo apt install g++-multilib make`
2. Run the command `make` in the `onnlab` folder and the build process should begin. The building may take several minutes. As a result, you should get an executable file `bin/experiment`.

3. Run the experiment using command `./bin/experiment`.

### Experiment Selection

Initially, the experiment `exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1` was selected to collect statistical data on the training of **ReLU-32**, **nmReLU-JN**, **nmiReLU-JN**, and **nmiReLU-EQ**, as well as on the performance of the **Carlini-Wagner L2** and **Basic Iterative Method** attacks against them. If you need exactly these statistics, you don't need to choose another experiment.

For each experiment, there is a separate compilable file that contains all the settings for that experiment. The main function called in the file (the experiment’s entry point) has the same name as the file, but without the extension. All callable experiment functions (entry points) are listed in the header file `onnlab/onnlab.h`. It is assumed that the experiment function will be called from the `main()` function, which is located in the main compilable file `onnlab/onnlab.cpp`.

Note that some compilable files are part of the framework rather than experiments - unlike experiment files, their name matches the name of the framework header file, differing only by the extension. You can distinguish an experiment file from a framework file by the `exp_` prefix in the experiment file's name.

## Experimental Data Processing

The raw data collected during the experiments conducted is located in the folder `mnist-results/raw_data`. Scripts for data processing in **Python** are located in folder `mnist-results/data_processing`.

The distribution of the collected metric data by configurations and logs is shown in the tables below. The data is divided into three tables for easier presentation.

Log file | Configuration | Experiment Id | Collected data
:------------ | :-- | :-: | :---
`exp_iReLU_ReLU_mnist_cmp2_relu.log` | ReLU-32 | [1] | accuracy, F1
`exp_iReLU_ReLU_mnist_cmp2_irelu.log` | iReLU-32 | [1] | accuracy, F1
`exp_nm1ilReLU_ReLU_mnist_cmp1_ReLU.log` | ReLU-64 | [2], [3] | accuracy, F1
`exp_nm1ilReLU_ReLU_mnist_cmp1_nmiReLU-JN.log` | nmiReLU-JN | [3] | accuracy, F1, util, util2
`exp_nm1ilReLU_ReLU_mnist_cmp1_nmiReLU-EQ.log` | nmiReLU-EQ | [2] | accuracy, F1, util, util2
`exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1__ReLU.log` | ReLU-32 | [4] | accuracy, F1, util, util2, CWL2s, BIMs, CWL2i, BIMi
`exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1__nmReLU.log` | nmReLU-JN | [4] | accuracy, F1, util, util2, CWL2s, BIMs, CWL2i, BIMi
`exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1__nmiReLU_JN.log` | nmiReLU-JN | [4] | accuracy, F1, util, util2, CWL2s, BIMs, CWL2i, BIMi
`exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1__nmiReLU_EQ.log` | nmiReLU-EQ | [4] | accuracy, F1, util, util2, CWL2s, BIMs, CWL2i, BIMi


Experiment Id | Experiment source file
| -: | :----
[1] | `exp_iReLU_ReLU_mnist_cmp2.cpp`
[2] | `exp_nm1ilReLU_ReLU_mnist_cmp1.cpp`
[3] | `exp_nm1ilReLU_ReLU_mnist_cmp2.cpp`
[4] | `exp_nm1ilReLU_ReLU_mnist_CWL2_BIM_cmp1.cpp`

Collected data | Description
| -: | :--------------
accuracy | Accuracy metric (percent)
F1 | F1-score weighted metric
util | Non-monotonity utilization (in average): $$\frac{1}{N}\sum^{N}_{i=1} \frac{\operatorname{min}(n^{-}_{i},n^{+}_{i})}{\operatorname{max}(n^{-}_{i},n^{+}_{i})} \cdot 100\%$$ Here, $N$ is the number of neurons with a non-monotonic activation function in the entire model;$n_i^-$ is the total number of occurrences where the input to the activation function falls into the negative felid ($x_i < 0$) of neuron $n_i$; $n_i^+$ is the total number of occurrences where the input to the activation function falls into the positive felid ($x_i \geq 0$) of neuron $n_i$
util2 | Non-monotonity strong utilization (in average): $\frac{1}{N}\sum^{N}_{i=1} \frac{\operatorname{min}(n^{-}_{i},n^{+}_{i})}{\operatorname{max}(n^{-}_{i},n^{+}_{i})} \cdot 100\%$ - same as **util**, but here only for the fields where the gradient value is not equal to one. **util** counts values that fall into the positive or negative fields of the argument, even if the absolute value is small, whereas strong utilization (**util2**) considers only values with a sufficiently large magnitude, applying a stricter criterion: $n^{-}_{i} < -\beta \lor n^{+}_{i} > \beta$.
CWL2s | *Carlini-Wagner L2* attack success rate (percent)
CWL2i | *Carlini-Wagner L2* attack overall iterations count (the attack stopped if the *loss* did not show noticeable reduction during some period)
BIMs | *Basic Iterative Method* attack success rate (percent)
BIMi | *Basic Iterative Method* attack overall iterations count (the attack stopped if the *loss* did not show noticeable reduction during some period)

### Scripts Environment

The scripts for data processing are written in **Python 3** (the developer had Python 3.12). 

The packages are required for the scripts to work: `numpy`, `matplotlib`, `seaborn` (the developer had `numpy==1.26.4`, `matplotlib==3.8.4`, `seaborn==0.13.2`).

If you don't have the required packages installed, you can install them by executing the following command: `pip install numpy matplotlib seaborn`. Package versions are not specified, as fairly recent versions should work correctly.

Ensure the required `*.log` files are in place `../raw_data` relative to script location (work folder location).

Every script from `data_processing` will:
1. Parse the log files.
2. Print summary statistics (mean and standard deviation, records count) for each configuration.
3. Display a GUI Window with KDE (Kernel Density Estimation) plots for the selected series.

Notes:
- The script assumes all input files are well-formatted tab-separated text files.
- Any lines that cannot be parsed (e.g., due to formatting errors) will be silently skipped.
- The plotting code may need customization if another metrics are to be visualized.

### Processing Scripts

| Script name | Description |
-: | :-------
`train_accuracy.py` | This script works for all six configurations. It calculates the mean and standard deviation for *accuracy* and *F1* score for each, as well as their correlation coefficient. For monotonic configurations, the first column represents *accuracy* and the second represents *F1* score. The same applies to non-monotonic configurations, but they also include additional columns for *utilization* and *strong utilization*. The GUI presents three graphs comparing the accuracy of the various configurations.
`train_utilization.py` | This script works only for non-monotonic configurations. It calculates the mean and standard deviation for *accuracy*, *F1*, *util* (*utilization*), and *util2* (*strong utilization*) for each configuration. The GUI presents two graphs comparing the configurations - the first graph uses the *util* metric, and the second uses the *util2* metric.
`attacks.py` | This script works only for configurations: *ReLU-32*, *nmReLU-JN*, *nmiReLU-JN*, *nmiReLU-EQ*. It calculates the mean and standard deviation for *CWL2-success*, *CWL2-iterations*, *BIM-success*, *BIM-iterations* for each configuration. The GUI presents four graphs comparing the success rates over configurations: first row graphs for ReLU-32 comparison (on the "together" graph its metrics overlaps the others); the remaining graphs show a comparison for the remaining configurations (excluding *ReLU-32*).
`stat_p_calc.py` | This script uses the Mann–Whitney U test (also known as the Wilcoxon rank-sum test) to statistically compare performance and robustness metrics between different neural network configurations. For each comparison, it prints the U-value (test statistic) and the P-value, which indicates whether there is a statistically significant difference between the distributions. Specifically, the test is applied to compare *accuracy* between *nmiReLU-EQ and ReLU-32*, *CWL2 attack* success rates between *nmiReLU-EQ and ReLU-32*, and *BIM attack* success rates both between *nmiReLU-EQ and ReLU-32*, and between *nmiReLU-EQ and nmReLU-JN*.

## Framework Documentation

### Key Interfaces

- **NeuronBasicInterface (NBI)** - `NNBasicsInterfaces.h`  
  Base interface for any class representing a neuron. A neuron must implement an activation function and input summation. All components assume neurons inherit from this interface.

- **ConnectionBasicInterface (CBI)** - `NNBasicsInterfaces.h`  
  Base interface for any directed connection between neurons. A connection must have a weight. All components assume connections inherit from this interface.

- **BasicWeightOptimizableInterface** - `BasicWghOptI.h`  
  Interface for connections that support weight optimization via optimizers (e.g., Adam). Abstracts the user from the optimizer internals.

- **BasicBackPropogableInterface** - `BasicBackPropgI.h`  
  Interface for neurons capable of storing and accumulating error for backpropagation.

- **MaccBackPropogableInterface** - `BasicBackPropgI.h`  
  Extension of the above for neurons whose activation is surjective, allowing access to pre-activation sums.

- **ZeroGradBackPropogableInterface** - `BasicBackPropgI.h`  
  For neurons with zero-gradient zones, provides an alternative, possibly non-zero, backpropagation value.

- **BasicLayerInterface** - `BasicLayerI.h`  
  Interface for layers of neurons. Neurons within a layer should not be directly connected to each other.

- **BackPropMetaLayerMark** - `BasicBackPropgI.h`  
  Marker interface for meta-objects that are not standard neuron layers and are not suitable for direct propagation.

- **BatchNeuronBasicI** - `BatchNeuronBasicI.h`  
  Interface for neurons that support batch (vector) propagation. Default batch size is 1 if not implemented.

- **InputNeuronI** - `InputNeuronI.h`  
  Interface for input-type neurons. Must not have incoming connections. Outputs are externally set.

- **OptimizerI** - `OptimizerI.h`  
  Interface for weight optimizers. Implementations define a `State` type and manage learning rate and hyperparameters.

- **ErrorCalculatorI** - `ErrorCalculatorI.h`  
  Interface for computing the error derivative (loss gradient) between predicted and target values.

- **CustomBackPropogableInterface** - `CustomBackPropgI.h`  
  For neurons with custom error-routing logic, such as selecting which parallel output path receives the gradient.

- **SelectableInputInterface** - `SelectableInputI.h`  
  For neurons that may output NaN as a signal. These neurons can be skipped during backpropagation.

- **BasicConvolutionI** - `BasicConvolutionI.h`, `BasicConvolutionI.cpp`  
  Interface for convolution operations. Typically, these are not neurons but algorithmic modules that perform forward propagation and delegate backpropagation to output buffers.

- **BasicConvolutionEssenceI** - `BasicConvolutionI.h`  
  Defines convolution parameters and maps convolution indices to neurons. Responsible for structure and geometry.

- **LimitedNeuronI** - `LimitedNeuronI.h`  
  Interface for neurons with limited input or output ranges (e.g., sigmoid).

- **Monotonic2FieldsHeuristicsI** - `Monotonic2FieldsHeuristicsI.h`  
  Interface for projecting weights between neuron groups (2-to-1 or 1-to-2) used in Equivalent Exchange heuristics.

### Connections (Implementations of ConnectionBasicInterface)

- **NNB_Connection** - `NNB_Connection.h`  
  Standard trainable directed connection.

- **NNB_StraightConnection** - `NNB_StraightConnection.h`  
  Fixed-weight, non-trainable directed connection.

- **NNB_Connection_spyable** - `NNB_Connection_spyable.h`  
  Debuggable connection that stores the last gradient value.

- **NNB_ConnWghAverager** - `NNB_ConnWghAverager.h`  
  Utility to externally average multiple connection weights.

- **HyperConnection** - `HyperConnection.h`  
  Creates multiple connections with a shared, synchronized weight. Gradients are averaged.

### Connection Groupings

- **DenseLayerStaticConnectomHolder** - `DenseLayerStaticConnectomHolder.h`  
  Creates and holds fully connected layer connections.

- **SparceLayerStaticConnectomHolderOneToOne** - `SparceLayerStaticConnectomHolderOneToOne.h`  
  Creates one-to-one connections between layers.

- **SparceLayerStaticConnectomHolder2Mult** - `SparceLayerStaticConnectomHolder2Mult.h`  
  Creates two-to-one connections using "half" or "alternating" schemes.

### Neurons with Linear Activation

- **NNB_Input** - `NNB_Input.h`  
  Input neuron reading from a pointer (or batch of pointers).

- **NNB_ConstInput** - `NNB_ConstInput.h`  
  Constant-output input neuron. Useful as a bias source.

- **NNB_Input_spyable** - `NNB_Input_spyable.h`  
  Input neuron that supports training and error accumulation.

- **NNB_Storage** - `NNB_Storage.h`  
  Input-like neuron that receives values programmatically and can accumulate error. Often used for convolution output.

- **NNB_Linear** - `NNB_Linear.h`  
  Linear neuron with adjustable (but non-trainable) offset and scale.

- **NNB_LinearSlim** - `NNB_LinearSlim.h`  
  Like `NNB_Linear`, but uses template parameters for offset/scale, which can be disabled.

### Typical Neurons

- **NNB_Sigmoid** - `NNB_Sigmoid.h`  
  Sigmoid activation neuron.

- **NNB_ReLU** - `NNB_ReLU.h`  
  Leaky ReLU neuron with required leak coefficient.

- **NNB_ReLU0** - `NNB_ReLU0.h`  
  Standard ReLU (subject to "dying ReLU" issue).

### Layer Groupings

- **NNB_Layer** - `NNB_Layer.h`  
  Standard neuron layer implementing `BasicLayerInterface`.

- **NNB_LayersAggregator** - `NNB_LayersAggregator.h`  
  Groups entire layers instead of individual neurons.

- **NeuronHoldingStaticLayer** - `NeuronHoldingStaticLayer.h`  
  Creates and stores neurons of the same type within a layer.

### Convolutions

- **NNB_ConvolutionHead** - `NNB_ConvolutionHead.h`  
  Standard convolution with trainable weights and optional bias.

- **NNB_ConvolutionMinMaxPoolingHead** - `NNB_ConvolutionMinMaxPoolingHead.h`  
  Implements MaxPooling and MinPooling.

- **NNB_ConvolutionEssence1d** - `NNB_ConvolutionEssence1d.h`  
  One-dimensional convolution essence with stride, dilation, and multi-channel output.

- **NNB_ConvolutionEssence2d** - `NNB_ConvolutionEssence2d.h`  
  Two-dimensional rectangular convolution essence.

- **NNB_ConvolutionEssence3d** - `NNB_ConvolutionEssence3d.h`  
  Three-dimensional cubic convolution essence.

- **NNB_ConvolutionEssence4d** - `NNB_ConvolutionEssence4d.h`  
  Four-dimensional convolution essence (tesseract). Can serve as a base for higher-dimensional cases.

### Optimizers

- **GradientDescendent** - `OptimizerGD.h`  
  Basic gradient descent optimizer. Only learning rate is configurable.

- **Adam** - `OptimizerAdam.h`  
  Implements Adam optimizer with full control over learning rate, beta1, beta2, and epsilon.

### Learning Providers (Training / Inference)

- **LearnGuiderFwBPg** - `LearnGuiderFwBPg.h`  
  Implements forward propagation, backpropagation (with batch averaging), weight optimization, and loss evaluation. Single-threaded.

- **LearnGuiderFwBPgThreadAble** - `LearnGuiderFwBPgThreadAble.h`  
  Multithreaded version of the above.

### Equivalent Exchange Tools

- **Monotonic2FieldsProjectingAccessory** - `Monotonic2FieldsProjectingAccessory.h`  
  Projects weights between neuron groups for Equivalent Exchange. Accepts two layers and the exchange strategy ("half" or "alternating").

- **Monotonic2FieldsHeuristicsEqExV1** - `Monotonic2FieldsHeuristicsEqExV1.h`  
  Projects weights using 2-to-1 and 1-to-2 mapping with $spread^{v1}$ metric.

- **Monotonic2FieldsHeuristicsEqExV2** - `Monotonic2FieldsHeuristicsEqExV2.h`  
  Similar to V1, but uses $spread^{v2}$ metric.

### Custom ReLU Variants (Implement NeuronBasicInterface)

- **NNB_iReLU** - `NNB_iReLU.h`  
  "Immortal ReLU" neuron.

- **NNB_ilReLU** - `NNB_ilReLU.h`  
  Immortal, limited ReLU.

- **NNB_nm1ReLU** - `NNB_nm1ReLU.h`  
  Non-monotonic Leaky ReLU.

- **NNB_nm1iReLU** - `NNB_nm1iReLU.h`  
  Non-monotonic immortal ReLU.

- **NNB_m1h_sqReLU** - `NNB_m1h_sqReLU.h`  
  Quadratic activation neuron (limited practical use).

- **NNB_nm1h_nanReLU** - `NNB_nm1h_nanReLU.h`  
  LeakyReLU with upper output limit triggering NaN. Implements `SelectableInputInterface`.

- **NNB_nm1h_SelectorHead** - `NNB_nm1h_SelectorHead.h`  
  Neuron that selects one of its input values as output. Implements `CustomBackPropogableInterface`.

### Adversarial Attacks

- **CarliniWagnerL2** - `CarliniWagnerL2.h`  
  Carlini-Wagner L2 attack (single-threaded). Configurable as in original paper.

- **CarliniWagnerL2ThreadAble** - `CarliniWagnerL2.h`  
  Multithreaded version of the above.

- **BasicIterativeMethod** - `BasicIterativeMethod.h`  
  Implements BIM/PGD attack. Includes loss-convergence early stopping. Single-threaded.

- **BasicIterativeMethodThreadAble** - `BasicIterativeMethod.h`  
  Multithreaded version of the above.

- **ReverseGuiderB2** - `ReverseGuiderB2.h`  
  Simplified BIM implementation with integrated training logic.

- **ReverseB1** - `ReverseB1.h`  
  Attack using reverse data propagation and linear system solver with efficient N-choose-K enumeration.

### Miscellaneous Utilities

- **F1scoreMulticlassWeightsGlobal** - `NetQualityCalcUtils.h`  
  Computes weighted and per-class F1 scores, as well as accuracy.

- **ClassWeightsCalculator** - `NetQualityCalcUtils.h`  
  Calculates class supports (weights) from dataset statistics.

- **CSVreader** - `CSVreader.h`  
  Reads CSV data as strings for parsing.

- **AtomicSpinlock** - `AtomicSpinlock.h`  
  Implements atomic spinlock for short waits.

- **InertAccumulator** - `InertAccumulator.h`  
  Floating-point accumulator with adjustable decay rate for older values.

- **IterableAggregation** - `IterableAggregation.h`  
  Utility to combine iterables of different types into a common base, usable with `for each` loops in C++.
