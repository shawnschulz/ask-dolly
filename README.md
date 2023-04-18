This is very much a WIP I use to mess around with databricks' newly released dolly 7b paramater model. You can either specify a directory containing a context.txt file with --memory_dir as well as a model_dir containing a saved pretrained model using the .save_pretrained, or use the --from_online flag to download it from dolly's huggingface repo. Note that the 7b paramter model is ~24gb and requires ~12gb of VRAM to run inference, possible you can get around that by tweaking how the model loads. If you use --from_online I recommend also specifying a model_dir so the script saves your model with updated weights (although tbh I'm not actually sure if it updates the weights automatically lol). 

The only thing I've really added so far is some code to prompt dolly with context from a context.txt file located in the repo directory, might update this in the future to use a proper json file rather than just appending text to the same text file.  

clone the repo: 

```
git clone https://github.com/shawnschulz/ask-dolly.git
```

cd into the cloned repo and run 

```
pip install -r requirements
```

If running from online, first time you run:

```
python3 ask_dolly.py --from_online --memory_dir /path/to/ask-dolly --model_dir /path/to/model_dir --prompt "Hello! How are you?"  
```

Otherwise run:

```
python3 ask_dolly.py --memory_dir /path/to/ask-dolly --model_dir /path/to/model_dir --prompt "Hello! How are you?" 
```

I also recommend adding an alias to your .bashrc for convenience:
```
#add this to your .bashrc!
alias 'python3 /path/to/ask-dolly/ask_dolly.py --memory_dir /path/to/ask-dolly --model_dir /path/to/model_dir --prompt '
```
To-do's:

- Make some proper files to save the memory dataset
- Test what works best for prompting it based on memory (enclose the prompt in brackets?)
- Add some kind of interactive mode to keep the the model loaded in memory 
- Experiment with other ways to instruction fine-tune the model

Link to databricks repo:
https://huggingface.co/databricks/dolly-v2-7b
 
