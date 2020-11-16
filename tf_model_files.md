## 工具开发者指南：TensorFlow 模型文件

大部分用户不需要关心 TensorFlow 如何在磁盘上存储数据的内部细节，但如果你是一个工具开发者，就不一样了。比如，你可能想要分析模型，或者在 TensorFlow 格式和其它格式之间来回转换。本指南试图解释如何处理保存模型数据的主文件的一些细节, 以使开发这些工具变得更容易。

## 协议缓存（Protocol Buffers）

所有的 TensorFlow 文件格式都是基于 [Protocol Buffers][protobuf] ，所以正式开始之前有必要先熟悉一下它的工作原理。简单来说，你用文本文件定义数据结构，然后使用 protobuf 工具生成 C、Python 和其它语言中的类，这样，开发者就可以用一种友好的方式加载、保存和访问数据。我们经常将协议缓存（Protocol Buffers）称为 protobufs，本指南会一直使用该约定。

## GraphDef 类

The basis of TensorFlow calculations is the Graph object, called a calculation graph. The calculation graph has a network of nodes, each node represents an operation, connected to each other as input and output. After creating a Graph object, you can save it by calling as_graph_def(), which returns a GraphDef object.

The GraphDef class is an object created by the ProtoBuf library according to the definition of tensorflow/core/framework/graph.proto. The protobuf tool analyzes this text file and generates code for loading, storing, and manipulating calculation graph definitions. If you see a single TensorFlow file that represents a model, it most likely contains a serialized version of a GraphDef object, and it was saved with protobuf code.

This generated code is used to save and load GraphDef files on disk. The actual code to load the model is as follows:：

```python
graph_def = graph_pb2.GraphDef()
```

This line creates an empty GraphDef object, which is a class created from the text definition in graph.proto. Next we will use this object to read data from the file.
```python
with open(FLAGS.graph, "rb") as f:
```

Here, a file handle is obtained according to the script parameters

```python
  if FLAGS.input_binary:
    graph_def.ParseFromString(f.read())
  else:
    text_format.Merge(f.read(), graph_def)
```

## Text or binary?

ProtoBuf actually supports two different file saving formats. TextFormat is a human-readable text form, which is very convenient for debugging and editing, but it becomes very large when storing numerical data, such as our common weight data. For a small example of this format, see [graph_run_run2.pbtxt][graph_run].

Compared to the text format, the binary format file will be much smaller, the disadvantage is that it is not easy to read. In the script, we will ask the user to provide a flag indicating whether the input file is binary or text, and then call the corresponding function based on this flag. You can find an example of a larger binary file `inception_v3_2016_08_28_frozen.pb` in [inception_v3 archive][inception_v3].

The API itself may be a bit confusing-the call to the binary format is actually ParseFromString(), and the loading of the text format uses a utility function in the text_format module.


## Node

Once the file is loaded into the graph_def variable, you can now access the data in it. For most practical applications, the important part is the node list stored in the node member. The following code demonstrates how to traverse these nodes:

```python
for node in graph_def.node
```

Each node is a NodeDef object, defined in [tensorflow/core/framework/node_def.proto][nodedef]. They are the cornerstone of constructing TensorFlow calculation graphs. Each node defines a simple operation and input connection. Below are the members of NodeDef and their descriptions.

### `name`

Each node should have a unique identifier, and it cannot conflict with other nodes in the calculation graph. If you do not specify a name when constructing a calculation graph with the Python API, TensorFlow will use the default name that reflects the type of operation, such as "MatMul", and then add a monotonically increasing number, such as "5", as the node name. This name will be used in some occasions, such as connecting nodes, or setting input and output when the calculation graph is running.

### `op`

op defines the operation to be run, for example, "Add", "MatMul", or "Conv2D". When a calculation graph is run,
This operation name is used to find its specific implementation in the TensorFlow registry. This registry is obtained by calling the macro REGISTER_OP, similar to
[tensorflow/core/ops/nn_ops.cc][nnops].

### `input`

input is a list of strings, where each string is the name of another node, optionally followed by a colon and the output port number of that node. For example, if a node has two inputs, this list is similar to ["some_node_name", "another_node_name"], which is equivalent to ["some_node_name:0", "another_node_name:0"], which means the first One input is the first output of the node named "some_node_name", and the second input is the first output of the node named "another_node_name".

### `device`

device represents the device used by the node. In most cases, you can ignore this because it is mainly for distributed environments.
Or it will be used when you force it to run on the CPU or GPU.

### `attr`

attr is a dictionary data structure that stores all the attributes of a node by key/value. They are the permanent properties of the node, that is, they will not change at runtime, such as the size of the convolution filter, or the value of the constant operation. Because there are so many types of attribute values, from strings, to integers, to arrays of tensor values, etc., a special protobuf file is needed to define the data structure for storing these attributes. For details, please refer to [tensorflow/core /framework/attr_value.proto][attr_proto].

Each attribute has a unique name string, and the expected attributes are listed when the operation is defined. If an attribute does not exist in the node,
But it lists the default value in the operation definition, and the default value will be used when creating the calculation graph.

In Python, you can access all these members by calling node.name, node.op and other methods. The node list stored in GraphDef constitutes a complete definition of the computational graph model framework.

## Freezing

Confusingly, the weights during training are usually not stored in the above file format. Instead, they are saved in a separate checkpoint file, and the calculation graph contains some Variable operations (op), which are used to load the values ​​in the latest checkpoint file during initialization. But when deploying to a production environment, it is not very convenient to use a separate file, so there is a script freeze_graph.py. Its function is to freeze a calculation graph definition and some checkpoint files into a single file.

In this process, the script will load GraphDef first, and then extract the values ​​of those variables from the latest checkpoint file.
Then replace each Variable operation with a Const operation, at this time the weight is stored in its attribute.
After that, all redundant nodes not related to forward reasoning will be eliminated, and the final GraphDef will be output to a file.

## Weight Formats

If you want to use a TensorFlow model to represent a neural network, one of the most common problems is how to extract and understand the weights. The common storage method is to use the freeze_graph script to store the weights as Tensors in the Const operation. These weights are defined in [tensorflow/core/framework/tensor.proto][tensor_proto]
, Which contains the data size and type, as well as the values ​​themselves. In Python, by calling operations such as `some_node_def.attr['value'].tensor`, a TensorProto object can be obtained from the NodeDef representing the Const operation.

This will result in an object representing weight data. The data itself is stored in one of the lists whose name is suffixed with `_val`. The name of the list reflects the type of this object. For example, float_val represents a 32-bit floating point data type.

When converting between different frameworks, the storage order of the convolutional layer weights is often a bit unpredictable. In TensorFlow, the filter weights of the two-dimensional convolution Conv2D operation are stored on the second input in the order of
`[filter_height, filter_width, input_depth, output_depth]`, where `filter_count` increased by 1 means moving to the next adjacent value in memory.

I hope that through such an overview, you can better understand the internal details of TensorFlow model files. If you need to manipulate these model files one day, I hope this article can be helpful.
[原文][source]

[protobuf]: https://developers.google.com/protocol-buffers
[graph_proto]: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto
[graph_run]: https://github.com/tensorflow/tensorboard/blob/master/tensorboard/demo/data/graph_run_run2.pbtxt
[inception_v3]: https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz
[nodedef]: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/node_def.proto
[nnops]: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/nn_ops.cc
[attr_proto]: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/attr_value.proto
[tensor_proto]: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto
[source]: https://www.tensorflow.org/extend/tool_developers/
