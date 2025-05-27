# Hello World Example

This sketch is the version adapted to the **MicroTFLite library** of the classic _hello_world_ example included in the official **TensorFlow Lite Micro** repository. Its purpose is to provide an example of the structure of a typical classification program that uses a pre-trained neural network model, whose _inference_ algorithm (i.e. input classification) is executed using the **TensorFlow Lite Micro library**.

The sketch uses the model registered in the official **TFLite-micro repository**, trained to compute the values of the function _y=sin(x)_. The model used in this sketch is the original _non-quantized_ version.
The **MicroTFLite** library handles models where values can be represented with _float_ numbers or in a quantized form.
The equivalent version of this example, included in the **Chirale_TensorFlowLite** library, uses the _quantized_ version of the same model. The non-quantized version typically requires more processing and memory resources but is slightly more accurate. The **MicroTFLite** library is not designed to manage optimizations and fine-tuning of resources. Experienced users can use the **Chirale_TensorFlowLite** library to access all optimization and fine-tuning features offered by **TensorFlow Lite Micro**.

The binary file in _.tflite_ format, containing the trained model, has been transformed into a constant char array so that it can be loaded into memory directly at compile time. In TensorFlow Lite applications developed on devices with an external file system, such as smartphones and single-board computers (e.g., Raspberry Pi), the model is stored on the external disk or SD Card as a binary file with a _.tflite_ extension and subsequently loaded into memory at runtime during the initialization phase. To enable Arduino boards, which lack an external file system, to use the model, the contents of the .tflite file have been included within an array that is compiled along with the code. This array, named _model_, is declared and initialized in **model.h** file.

The **neural network model** trained to compute the _sin(x)_ function is a **DNN** with 16 neurons in the first layer (ReLU activation function), 16 neurons in the second layer (hidden layer with ReLU activation), and a single neuron in the output layer. The single float value representing the input is connected to all neurons of the first layer.
For more information on how this model was built and trained, you can refer to the [official TensorFlow repository](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world "Repository tflite-micro").

The comments within the sketch code explain each step. The relevant steps are:

1. Include the library header and the model.h file containing the model in the .tflite format
2. Define the _Tensor Arena_ size and allocate it as a global byte array. The **tensor arena** is a contiguous block of memory allocated to store the tensors (inputs, outputs, and intermediate results) required by a TensorFlow Lite Micro model during inference. In the context of microcontrollers and other resource-constrained devices, managing memory efficiently is crucial, and the tensor arena plays a vital role in this. The entire tensor arena is allocated once, usually at the start of the program, and it is used to store all tensors throughout the inference process. This reduces the complexity and potential issues related to memory allocation and deallocation.
3. Init the TensorFlow Lite Micro environment, calling the **ModelInit** function in the setup() Arduino function. **ModelInit()** receives, as parameters, the _model_ array, the _tensor arena_ array and its size. **ModelInit** performs all initialization operations and allocate the input and output tensors. The size of these tensors are defined inside the model. In this case, the input and the output are single float numbers, so the size of input tensor is 1 and the size of output tensor is 1.
4. Wait for an user input from the serial monitor terminal.
5. Process the input provided, assuring that it is a float number between 0 and 2\*Pi
6. Load the input value in the first (and only) position of the input tensor, calling the **ModelSetInput** function. The first position has index 0.
7. Run inference calling the **ModelRunInference** function.
8. Read the output value from the first, and only, position of the output tensor, calling the **ModelGetOutput** function. The first position has index 0.
9. Write output value evaluated by the model and the actual sin(x) value, evaluated by the regular math function, to serial monitor.
