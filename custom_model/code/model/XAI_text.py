from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=['real','fake'])

def lime_explain_text(sample_text, image_tensor):
    # we fix the image_tensor; treat text as variable
    def wrapped_predict(text_list):
        # convert each text in text_list to input_ids/attention_mask
        # replicate the same image_tensor for each
        batch_ids, batch_mask = tokenizer(text_list, …)
        probs = predict_fn(batch_ids, batch_mask, image_tensor.repeat(len(text_list),1,1,1))
        return probs
    exp = explainer.explain_instance(sample_text, wrapped_predict, num_features=10)
    return exp.as_list()