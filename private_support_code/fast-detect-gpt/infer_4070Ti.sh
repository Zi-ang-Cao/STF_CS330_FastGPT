Folder="4070Ti"
# ModelName="S_0_RP_True_SC_True_SPC_True_1203_180451_1"
ModelName="S_1_RP_True_SC_True_SPC_True_1203_233532_1"




## With RuleBased
python3 INFER.py \
    --overfit_sentence_idx 1 \
    --Infer_modelName ${ModelName} \
    --Infer_folder ${Folder} \



######## ID = 0 ########

# _, perturbed_output = self._perturb_tokens(self.last_logits, perturb_mode="chosen", perturb_ranking=4) # 3
# [RL]
#         RL_Score=0.01
#         RL_Text=Mark Mercer, 47, of Toll Bar Houses, Workington, is accused of taking??1,530 from Maryport Post Office on Monday as well as several firearms offences. No pleas were entered during a brief hearing at Carlisle Magistrates' Court and Mr Mercer was remanded in custody. A 24-year-old woman also arrested in connection with the raid has been released on bail until 14 March.  A police spokesman said officers were "aware of the incident" on Monday but had not made contact yet with the suspect


# [Raw]
#         Raw_Score=0.06
#         Raw_Text=Mark Mercer, 47, of Toll Bar Houses, Workington, is accused of taking??1,530 from Maryport Post Office on Monday as well as several firearms offences. No pleas were entered during a brief hearing at Carlisle Magistrates' Court and Mr Mercer was remanded in custody. A 24-year-old woman also arrested in connection with the raid has been released on bail until 14 March. A 28-year-old woman known as Miss Elfin from Dickenbarger was questioned at Carlisle police station on Monday evening.


######## ID = 1 ########

# [RL]
#         RL_Score=0.86
#         RL_Text=The Old Royal Station in Ballater was ravaged by the blaze in May 2015. The old station had been the final stopping point for members of the Royal Family heading to Balmoral. A visitor information centre and a restaurant will feature in the new building along with a library and an enhanced exhibition space. The work is expected to be completed in December.  The Royal Bank has also committed a further $1m in funding to support the refurbished building
#  -------------------- 
# [Raw]
#         Raw_Score=1.0
#         Raw_Text=The Old Royal Station in Ballater was ravaged by the blaze in May 2015. The old station had been the final stopping point for members of the Royal Family heading to Balmoral. A visitor information centre and a restaurant will feature in the new building along with a library and an enhanced exhibition space. The work is expected to be completed in December.   The Old Royal Station was built in 1885 and was the last stop for the Royal Family before they left for Balmoral.
#  -------------------- 