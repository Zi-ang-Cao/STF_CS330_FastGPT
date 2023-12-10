# python3 GPT_A1.py --overfit_sentence_idx 0 \
#     --InStepScore --InStepPerturbCost \
#     --cross_Sentence \
#     --retrain_from ./PPO_prior/S_1_RP_True_SC_True_SPC_True_1203_233532_T_300000.pt \

# python3 GPT_A1.py --overfit_sentence_idx 1 \
#     --InStepScore --InStepPerturbCost \
#     --cross_Sentence \
#     --retrain_from ./PPO_prior/S_0_RP_True_SC_True_SPC_True_1203_180451_T_300000.pt \


python3 GPT_A1.py --overfit_sentence_idx 0 \
    --InStepScore --InStepPerturbCost \
    --cross_Sentence \
    --ruleBasedPenalty \
    --retrain_from ./PPO_prior/S_1_RP_True_SC_True_SPC_True_1203_233532_T_300000.pt \

python3 GPT_A1.py --overfit_sentence_idx 0 \
    --InStepScore --InStepPerturbCost \
    --retrain_from ./PPO_prior/S_0_RP_True_SC_True_SPC_True_1203_180451_T_300000.pt \

python3 GPT_A1.py --overfit_sentence_idx 0 \
    --InStepScore --InStepPerturbCost \
    # --retrain_from ./PPO_prior/S_1_RP_True_SC_True_SPC_True_1203_233532_T_300000.pt \
