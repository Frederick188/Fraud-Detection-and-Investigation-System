from src.agent import FraudAgent

agent = FraudAgent()

while True:

    q = input("\nAsk: ")

    if q.lower() == "exit":
        break

    print("\n")
    print(agent.ask(q))