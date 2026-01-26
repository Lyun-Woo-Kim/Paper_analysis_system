from utils.utils import load_config
from llm_load.rag import PaperInfo_RAG
from utils.prompts import FIELD_SELECT_PROMPT, REFINE_JSON_PROMPT, RAG_ANSWER_PROMPT
from llm_load.app_manager import AppManager
from utils.utils import parse_json_output, delete_think_tag

if __name__ == "__main__": 
    app_manager = AppManager()
    app_manager.start_llm_server()
    
    configs = load_config("/rwkim/AI-DEV/IoT-S_xqbot/files/RLs/paper_analysis/model_config.yaml")
    rag_app = PaperInfo_RAG(configs, re_rank = True)
    llm_model = rag_app.load_fix_langchain_model()
    # question = "What is the main idea of the paper?"
    while True: 
        question = input("Enter your question: ")
        if question == "exit":
            break
        try: 
            _field = llm_model.invoke(FIELD_SELECT_PROMPT.format(question = question))
            _field = parse_json_output(_field.content)
        except Exception as e: 
            _field = llm_model.invoke(REFINE_JSON_PROMPT.format(broken_text = delete_think_tag(_field), error_msg=str(e)))
            _field = parse_json_output(_field.content)
        retriever_k_dict = {
            "text": 10,
            "equation": 2, 
            "visual": 2
        }
        main_chain = rag_app.build_rag_chain(
            prompt_template = RAG_ANSWER_PROMPT,
            retriever_k_dict = retriever_k_dict,
            collections = _field['collections']
        )
        try: 
            answer = main_chain.invoke({"question": question})
            answer = parse_json_output(answer)
        except Exception as e:
            answer = llm_model.invoke(REFINE_JSON_PROMPT.format(broken_text = delete_think_tag(answer), error_msg=str(e)))
            answer = parse_json_output(answer)
        print(answer)