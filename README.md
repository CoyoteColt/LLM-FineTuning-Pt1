# **Fine Tuning Of An LLM To Aid In Symptom Diagnosis**

In this project, we will learn how to Fine Tuning an AI model. We will use LLM Starling and make it an expert in the medical field, being able to guide and diagnose symptoms.
Due to its size, we will divide this project into two parts:
- **Part 1**: Everything related to the Fine-Tuning of the model and its deployment.
- **Part 2**: We will create a web system (very simple) so that anyone can consult their symptoms through a web page, very similar to ChatGPT

<img src="image/doctor-image.jpg" alt="doc_image" width="800">


### What is Fine-tuning an AI model?

Fine-tuning is the process of training an existing AI model, trained on a generic dataset, to adapt it to a new dataset or objective. This allows the model to be more effective at handling the specific characteristics of the new dataset. When an AI model is trained, it is tuned to learn to recognize patterns and relationships in the training data. However, if the model is trained on a dataset that is not directly related to the dataset you want to apply it to, it may not be as effective at handling the specific characteristics of the new dataset. This is where fine-tuning comes into play. Fine-tuning involves reusing the pre-trained model and adjusting it to adapt it to the new dataset. This is done by adjusting the model’s parameters, such as weights and biases, so that it is more effective at handling the specific characteristics of the new dataset.
Fine-tuning can be useful in a number of situations, such as:

**Knowledge transfer**: when a model trained on one dataset is applied to a new, related but not identical dataset.

**Parameter tuning**: when a pre-trained model needs to be fine-tuned to better adapt to the specific characteristics of the new dataset.

**Performance improvement**: when a pre-trained model needs to be fine-tuned to better adapt to the specific characteristics of the new dataset and improve its performance.

## DataSet Llama2 MedTuned Instructions
Llama2-MedTuned-Instructions is an instruction-based dataset developed for training language models in biomedical NLP tasks. It consists of approximately 200,000 samples, each tailored to guide models in performing specific tasks such as Named Entity Recognition (NER), Relation Extraction (RE), and Medical Natural Language Inference (NLI). This dataset represents a fusion of various existing data sources, reformatted to facilitate instruction-based learning.

**Source Datasets and Composition**

The dataset amalgamates training subsets from several prominent biomedical datasets:
- Named Entity Recognition (NER): Utilises NCBI-disease, BC5CDR-disease, BC5CDR-chem, BC2GM, JNLPBA, and i2b2-2012 datasets.
- Relation Extraction (RE): Incorporates i2b2-2010 dataset.
- Natural Language Inference (NLI): Employs the MedNLI dataset.
- Document Classification: Uses the hallmarks of cancer (HoC) dataset.
- Question Answering (QA): Includes samples from ChatDoctor and PMC-Llama-Instructions datasets.

**Prompting Strategy**

Each sample in the dataset follows a three-part structure: Instruction, Input, and Output. This format ensures clarity in task directives and expected outcomes, aligning with the instruction-based training approach.

**Usage and Application**

This dataset is ideal for training and evaluating models on biomedical NLP tasks, particularly those focused on understanding and processing medical and clinical text. It serves as a benchmark for assessing model performance in domain-specific tasks, comparing against established models like BioBERT and BioClinicalBERT.

**Link to DataSet**

If you want to look at the original dataset link for further explanation: [Llama2-MedTuned-Instructions](https://huggingface.co/datasets/nlpie/Llama2-MedTuned-Instructions)






## Quantization Parameters

Large Language Model (LLM) quantization is a process of reducing the precision of a large language model (LLM)'s parameters to a lower value without losing much of its performance. This is done to reduce the complexity of the model and consequently improve computational efficiency and storage efficiency.

LLM quantization is useful in several situations, such as:

- **Model Size Reduction**: Quantization can reduce the model size, which can improve data storage and transfer efficiency.

- **Improved Computational Efficiency**: Quantization can improve the computational efficiency of the model, making it faster and more efficient in terms of computational resources.

- **Increased Scalability**: Quantization can allow models to be trained on cheaper and more efficient hardware, which can be useful in production applications.

BitsAndBytesConfig is used to configure LLM quantization. It is an object that contains information about quantization, such as:

- **Quantization Type**: The type of quantization to use, such as floating-point quantization or integer quantization.

- **Number of Bits**: The number of bits to be used to represent the model parameters.

BitsAndBytesConfig is used to configure LLM quantization and can be used in conjunction with other parameters to train a more efficient and scalable language model.


## Large Language Model Starling-LM-7B-alpha

Starling-LM-7B-alpha is a language model trained from Openchat 3.5 with reward model berkeley-nest/Starling-RM-7B-alpha and policy optimization method advantage-induced policy alignment (APA). The evaluation results are listed below. An open large language model (LLM) trained by Reinforcement Learning from AI Feedback (RLAIF). The model harnesses the power of our new GPT-4 labeled ranking dataset, berkeley-nest/Nectar, and our new reward training and policy tuning pipeline. Starling-7B-alpha scores 8.09 in MT Bench with GPT-4 as a judge, outperforming every model to date on MT-Bench except for OpenAI's GPT-4 and GPT-4 Turbo. We released the ranking dataset Nectar, the reward model Starling-RM-7B-alpha and the language model Starling-LM-7B-alpha on HuggingFace, and an online demo in LMSYS Chatbot Arena. 

<img src="image\starling-rank.png" alt="starling-rank">

- **Developed by**: Banghua Zhu, Evan Frick, Tianhao Wu, Hanlin Zhu and Jiantao Jiao.
- **Model type**: Language Model finetuned with RLHF / RLAIF
- **License**: Apache-2.0 license under the condition that the model is not used to compete with OpenAI
- **Finetuned from model**: Openchat 3.5 (based on Mistral-7B-v0.1)

If you want to read the original documentation, follow the link: [Starling-LM-7B-alpha](https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha)

## LoRa Parameters for PEFT

PEFT (Parameter-Efficient Fine-Tuning) is an AI model fine-tuning method that aims to reduce the number of parameters in a model, making it more efficient in terms of computation and memory. This is especially useful when working with very large models or when you have limited computational resources.

PEFT is an approach that combines two concepts:
- **Pruning**: is the process of removing inactive or unimportant parameters from the model, reducing the number of parameters and, consequently, the complexity of the model.

- **Fine-tuning**: is the process of adjusting the model parameters to adapt it to a new data set.

PEFT works as follows:
- The pre-trained model is initialized with a set of parameters.

- The model is subjected to a pruning process, where the least important parameters are removed.

- The model is then fine-tuned to adapt it to the new dataset.

- The model is trained again, but now with a reduced set of parameters.

PEFT has several advantages, including:
- Reduced model complexity, which can improve scalability and computational efficiency.

- Reduced model size, which can improve data storage and transfer efficiency.

- Faster and more accurate model tuning, as the model is trained with a reduced set of parameters.

There are several ways to apply PEFT and one of them is **LoRa**.

LoRA (Low-Rank Adaptation of Large Language Models) is a fine-tuning method used in LLMs that aims to adapt these models to new datasets or tasks while maintaining the computational efficiency and accuracy of the original model.
LLMs are language models trained on large datasets and are capable of performing tasks such as text classification, text generation, and translation. However, these models can be very large and complex, which can make them difficult to adapt to new datasets or tasks.
This is where LoRA comes into play. LoRA is a fine-tuning method that aims to reduce the complexity of the model, making it more efficient in terms of computation and memory, while maintaining the accuracy of the original model.

LoRA works as follows:
- The LLM model is initialized with a set of parameters.

- Adapters are defined for a new set of parameters.

- The model is then fine-tuned and adapted to the new dataset or task.

- The model is retrained, but now with a smaller and more efficient set of parameters.

LoRA has several advantages, including:
- Reduced model complexity, which can improve scalability and computational efficiency.

- Reduced model size, which can improve data storage and transfer efficiency.

- Faster and more accurate model tuning, as the model is trained with a reduced set of parameters.

Here is the description of each parameter in the LoraConfig configuration:

- **r**: Specifies the size of the expansion factor used in the LoRA (Low-Rank Adaptation) mechanism. A value of r = 8 indicates that low-rank adaptation increases the dimensionality of the projections by a factor of 8. Essentially, this parameter controls the size of the dimensionality increase that is applied to selected layers of the model to allow for greater adaptability without significantly changing the total number of model parameters.

- **lora_alpha**: Multiplier for the learning factor applied specifically to the LoRA adaptation parameters. With lora_alpha = 16, the learning of these specific parameters is amplified by 16 times compared to the rest of the model parameters. This allows the adaptation parameters to adjust more quickly during training.

- **lora_dropout**: Dropout rate applied to LoRA modifications. A value of 0.05 means that 5% of the elements in the LoRA projections will be randomly zeroed out during training. Dropout is a regularization technique used to avoid overfitting by forcing the model to learn more robust representations since it cannot rely on any single connection between neurons.

- **bias**: Configures whether or not bias terms are used in LoRA adjustments. Here, it is set to "none", indicating that no bias terms will be added to LoRA adjustments. Choosing not to use bias can simplify the model and focus adjustments solely on the weights.

- **task_type**: Defines the type of task the model is being configured for. "CAUSAL_LM" refers to a causal language model, where the prediction of each subsequent word depends only on the previous words and not on any future words. This is typical of models that generate text in a sequential manner, such as text generation or dialogue modeling.

These parameters are configured to tune the model effectively and specifically, leveraging the low-rank adaptation technique to fine-tune the model without the computational cost of re-training all the parameters of the original model.



## Training Arguments

The parameters of the TrainingArguments object are used to configure the training process of a machine learning model. Here is the description of each of them:

- **output_dir**: Directory where the trained model artifacts, such as model weights and settings, will be saved. Here, it is set to "adjusted_model".

- **per_device_train_batch_size**: Number of examples processed simultaneously on each device (such as a GPU) during training.

- **gradient_accumulation_steps**: Number of training steps to accumulate gradients before performing a parameter update. This allows using a larger effective batch size than can fit in the device memory.

- **optim**: Optimizer used to update model weights. "paged_adamw_32bit" is a variation of the AdamW optimizer that optimizes memory usage by using 32-bit precision.

- **learning_rate**: Initial learning rate used by the optimizer.

- **lr_scheduler_type**: A type of learning rate scheduler, which adjusts the learning rate based on the number of epochs or steps. "cosine" indicates that the learning rate follows a cosine curve, gradually decreasing over epochs.

- **save_strategy**: Strategy to save the model during training. "epoch" means that the model will be saved at the end of each epoch.

- **logging_steps**: Number of training steps after which logs will be written. Here it is set to 10, meaning that training progress will be logged every 10 steps.

- **num_train_epochs**: Total number of training epochs. An epoch is one complete pass through the training data.

- **max_steps**: Maximum number of training steps to be executed. Useful to terminate training after a fixed number of steps, regardless of epochs.

- **fp16**: If enabled, enables the use of 16-bit floating point during training to reduce memory usage and possibly speed up computation. Here, it is set to True, indicating that it is enabled.


## Supervised Fine-tuning Trainer (SFTT) Parameters

Supervised Fine-Tuning Trainer (SFTT) is a method for training Artificial Intelligence (AI) models that aims to improve the performance of a pre-trained model on a specific dataset. SFTT is a supervised training approach that combines the fine-tuning technique with the supervised training technique.

The SFTT works as follows:
- The pre-trained model is initialized with a set of parameters.
- The model is subjected to a fine-tuning process, where the parameters are adjusted to adapt it to the specific dataset.
- The model is trained again, but now with a set of parameters adjusted for the specific dataset.
- The model is trained with a labeled dataset, where each sample is associated with a label.
- The model is trained to predict the labels of the data, improving the performance of the model in relation to the specific dataset.

SFFT has several advantages, including:
- Faster and more accurate model tuning, as the model is trained with a specific dataset.
- Increased model performance in relation to the specific dataset.
- Possibility to train pre-trained models for new datasets.


Here is the description of each parameter used when creating an SFTTrainer instance:

- **model**: Defines the machine learning model to be trained.

- **peft_config**: Configuration for efficient model parameter training (PEFT).

- **max_seq_length**: Maximum string length that the model can accept. It is set to 512, which means that each input (such as text) will be truncated or padded to this length to maintain a consistent length across inputs.

- **tokenizer**: The tokenizer is used to convert text into a format that the model can process (such as tokens or numbers). Here, a specific tokenizer is used, which must be compatible with the chosen model.

- **packing**: A boolean parameter that, when True, indicates that the trainer should attempt to efficiently pack input sequences to optimize memory usage and training performance.

- **formatting_func**: Function used to format input data before it is provided to the model.

- **args**: Additional training configuration parameters, grouped in training_arguments. These arguments define various settings such as learning rate, saving strategy, among others, which were described previously.

- **train_dataset**: Dataset used to train the model.

- **eval_dataset**: Dataset used to evaluate the model.


## Saving the model to disk

Remembering that this Fine-Tuning was done by Google Colab due to its ease of implementation with GPUs, it is possible to run this same project entirely on your machine with Windows OS but it will require a little patience and time just by making some adjustments to the code. This last step of part 1 was like this:
- **Saving the model**: We save both the model and tokenize it using the save_pretrained function
- **Setting the file**: We import a library to assist in the compression process as shutil and also define the path of the file that will be compressed.
- **Download**: And finally, we download the compressed file.

## Version

If you want to reproduce it in your environment, below you will find all the versions used to assist in development.

<img src="image\version.png" alt="version">

## Final Considerations PT1

This is the end of the first part of the project in which we focused on understanding how Fine-Tuning an AI model works on new data and how to make a generic LLM an expert in a certain area.

Be sure to check out part two: [Deploy with Web System](https://github.com/CoyoteColt/LLM-FineTuning-Pt2?tab=readme-ov-file)