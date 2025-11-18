import json
from pathlib import Path

OUTPUT_FILE = Path("data/eval/rbc_eval_set.jsonl")

known_questions = [
    ("k01", "How do I report a lost or stolen RBC credit card?",
     "Call 1-800-769-2512 immediately to report a lost or stolen RBC credit card. RBC will block the card and issue a replacement."),
    ("k02", "How can I reset my RBC Online Banking password?",
     "You can reset your RBC Online Banking password using the 'Forgot Password' option and verifying your identity."),
    ("k03", "How do I send an Interac e-Transfer using RBC?",
     "You can send an Interac e-Transfer through RBC Online Banking or the RBC Mobile app by selecting a contact and entering an amount."),
    ("k04", "What should I do if my RBC debit card is not working?",
     "Ensure the card is activated and not damaged; you may request a replacement card if needed."),
    ("k05", "How do I change my address with RBC?",
     "Update your address through RBC Online Banking under Personal Settings or by contacting RBC."),
    ("k06", "How can I increase my RBC credit card limit?",
     "Request a credit limit increase through RBC Online Banking or by contacting RBC."),
    ("k07", "What are the daily ATM withdrawal limits for RBC?",
     "RBC accounts have daily ATM withdrawal limits that vary by card type and account profile."),
    ("k08", "How do I set up direct deposit with RBC?",
     "Provide your RBC account, transit, and institution numbers to your employer."),
    ("k09", "How do I pay a bill using RBC Online Banking?",
     "Use Pay Bills in RBC Online Banking, select a payee, and enter a payment amount."),
    ("k10", "How do I view my RBC credit card statements?",
     "You can view statements through RBC Online Banking or the RBC Mobile app."),
    ("k11", "How do I activate my new RBC credit card?",
     "Activate through RBC Online Banking or by calling the activation number."),
    ("k12", "How do I stop a recurring payment on my RBC account?",
     "Contact the merchant or request assistance through RBC Online Banking."),
    ("k13", "How do I dispute a transaction on my RBC credit card?",
     "Dispute the transaction through RBC Online Banking or by contacting RBC."),
    ("k14", "How can I order a new RBC chequebook?",
     "Order cheques through RBC Online Banking or by contacting RBC."),
    ("k15", "How do I enable RBC alerts on my account?",
     "Enable alerts through RBC Online Banking or the RBC Mobile app."),
    ("k16", "How do I update my RBC email address?",
     "Update your email under Personal Information in RBC Online Banking."),
    ("k17", "What should I do if I suspect fraud on my RBC account?",
     "Contact RBC immediately and review recent transactions."),
    ("k18", "How do I deposit a cheque using the RBC Mobile app?",
     "Use the Cheque Deposit feature in the RBC Mobile app."),
    ("k19", "How do I transfer money between my RBC accounts?",
     "Use RBC Online Banking or the RBC Mobile app."),
    ("k20", "How can I view my RBC account balance?",
     "View your balance using RBC Online Banking or the RBC Mobile app."),
]

unknown_questions = [
    ("u01", "How do I open a chequing account with TD Bank?"),
    ("u02", "What are the Scotiabank student account fees?"),
    ("u03", "Does CIBC offer cryptocurrency trading?"),
    ("u04", "What is the interest rate on a Bank of America Rewards card?"),
    ("u05", "How can I close my Chase Sapphire credit card?"),
    ("u06", "How do I apply for a Wells Fargo auto loan?"),
    ("u07", "What are the hours for the branch inside Walmart?"),
    ("u08", "What is the weather in Toronto today?"),
    ("u09", "Can I use my RBC debit card on Mars?"),
    ("u10", "How do I get a refund for a Steam purchase?"),
]

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, "w") as f:
        # Write known Q/A
        for qid, q, ans in known_questions:
            f.write(json.dumps({
                "id": qid,
                "type": "known",
                "question": q,
                "answer": ans
            }) + "\n")
        
        # Write unknown Q (no answer)
        for qid, q in unknown_questions:
            f.write(json.dumps({
                "id": qid,
                "type": "unknown",
                "question": q,
                "answer": None
            }) + "\n")
    
    print(f"Created: {OUTPUT_FILE}")
    print("Total lines:", len(known_questions) + len(unknown_questions))

if __name__ == "__main__":
    main()
