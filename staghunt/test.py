from Bishop import *

Observer = LoadObserver("StagHunt_a_PlayerA")
#Observer = LoadObserver("Tatik_T1_L1")
Results = Observer.InferAgent(
    ActionSequence=['R','R'], Samples=1000, Feedback=True)

print("PREDICTING NEXT ACTION(S)")
Probs = Observer.PredictAction(Results)
for plan,prob in list(zip(Probs[0],Probs[1])):
   print(plan,prob)
