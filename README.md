# automat-llm
Mobile AI Assistant

## How To Run
To run this demo first ust `pip install -r requirements.txt` to install basic packages from pip. Then you will have to optionally run `pip install ./dia` for voice interaction if desired.
the use of dia is entirely optional and requirements should demonstrate the basic demo. `python main.py` should be enough to get the demo going!

## Known Issues
Currently the demo has a known bug in which it's Inference Size is too small for the VectorStore/Retrieval Chain. We are currently working to circumvent this with the basic demo. That being said there are alternative functions available such as `create_rag_chain_mistral`, `create_rag_chain_falcon`, and `create_rag_chain_mixtral` that circumvent these issues though they subject the user to having to download ~14gbs+ LLM.
