import tensorflow as tf

# Define the paths to the pretrained models
model_paths = [
    r"C:\Users\HP\Downloads\PycharmProjects\hearing_aid_DTLN\pretrained_model\model_1.tflite",
    r"C:\Users\HP\Downloads\PycharmProjects\hearing_aid_DTLN\pretrained_model\model_2.tflite"
]

# Iterate over each model
for i, model_path in enumerate(model_paths, 1):
    print(f"\nExtracting features from model_{i}.tflite:")

    # Load and allocate the interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Extract and print input features
    print(f"\nInput Features ({len(input_details)} inputs):")
    for input_tensor in input_details:
        print(f"  - Tensor Name: {input_tensor['name']}")
        print(f"    Shape: {input_tensor['shape']}")
        print(f"    Data Type: {input_tensor['dtype']}")
        print(f"    Index: {input_tensor['index']}")
        if 'quantization' in input_tensor and input_tensor['quantization'][0] != 0.0:
            print(f"    Quantization: Scale={input_tensor['quantization'][0]}, ZeroPoint={input_tensor['quantization'][1]}")

    # Extract and print output features
    print(f"\nOutput Features ({len(output_details)} outputs):")
    for output_tensor in output_details:
        print(f"  - Tensor Name: {output_tensor['name']}")
        print(f"    Shape: {output_tensor['shape']}")
        print(f"    Data Type: {output_tensor['dtype']}")
        print(f"    Index: {output_tensor['index']}")
        if 'quantization' in output_tensor and output_tensor['quantization'][0] != 0.0:
            print(f"    Quantization: Scale={output_tensor['quantization'][0]}, ZeroPoint={output_tensor['quantization'][1]}")

    # Attempt to extract metadata (optional)
    try:
        from tflite_support.metadata import metadata_extractor
        extractor = metadata_extractor.MetadataExtractor(model_path)
        metadata = extractor.get_metadata()
        print("\nMetadata Features:")
        print(f"  - Model Name: {metadata.name}")
        print(f"  - Description: {metadata.description}")
        print(f"  - Version: {metadata.version}")
    except ImportError as e:
        print("\nMetadata Features: Not available (install tflite-support with 'pip install tflite-support' to enable)")
    except Exception as e:
        print(f"\nMetadata Features: Not available or error occurred: {str(e)}")