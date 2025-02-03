# Fuzzy-Human-Action-based-Emotion-Recognition

This is a emotion recognition through gesture and action series data, by using fuzzy tool to enhance the stgcn backbone model.

The train.py and test.py are in the tools.

For this method, its main improvement is in the head definition used by j.py config file. So "pyskl/models/recognizers/recognizergcn.py" is revised in "forward_train()", and a new head "MemConstGCNHead" is created under "pyskl/models/heads/MemConstGCNHead.py". Corresponding revision is also done in the "cls_head" part in "j.py" config file.
In this version, the "MemConstGCNHead" just take the original output with no modification which is transferred to "forward_train()" function in above "recognizergcn.py" directly. The revision of adding fuzzy tools are mainly within "forward_train()" function, where it adds fuzzy similarity relationships to enhance cross-entropy loss only optimization.

For data, it uses AFEW-VA in this version. Considering data size saving, it doesn't upload the raw image files here, but provide a link to the data just in the location they are.
