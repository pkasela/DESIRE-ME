# DESIRE-ME: Domain-Enhanced Supervised Information REtrieval using Mixture-of-Experts

Abstract:
Open-domain question answering requires retrieval systems able to cope with the diverse and varied nature of questions, providing accurate answers across a broad spectrum of query types and topics. To deal with such topic heterogeneity through a unique model, we propose DESIRE-ME, a neural information retrieval model that leverages the Mixture-of-Experts framework to combine multiple specialized neural models. We rely on Wikipedia data to train an effective neural gating mechanism that classifies the incoming query and that weighs correspondingly the predictions of the different domain-specific experts. This allows DESIRE-ME to specialize adaptively in multiple domains. Through extensive experiments on publicly available datasets, we show that our proposal can effectively generalize domain-enhanced neural models. DESIRE-ME excels in handling open-domain questions adaptively, boosting by up to 12% in NDCG@10 and 23% in P@1, the underlying state-of-the-art dense retrieval model.

How to recreate the dataset:
