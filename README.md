# automat-llm
Mobile AI Assistant

## How To Run
To run this demo first ust `pip install -r requirements.txt` to install basic packages from pip. Then you will have to optionally run `pip install ./dia` for voice interaction if desired.
the use of dia is entirely optional and requirements should demonstrate the basic demo. 
Before running the demo be sure to run `huggingface-cli login` and login as needed, then run
`python main.py`, it should work. If any Weaviate issues come up contact Sasori Zero Labs (mileslitteral@sasorizerolabs.com)

## Known Issues
Currently the demo has a known bug in which it's default model's Inference Size is too small for the VectorStore/Retrieval Chain. We are currently working to circumvent this with the basic demo. That being said there are alternative functions available such as `create_rag_chain_mistral`, `create_rag_chain_falcon`, and `create_rag_chain_mixtral` that circumvent these issues though they subject the user to having to download ~14gbs+ LLM. replacing `create_rag_chain` in automat_llm.config will fix this issue.
