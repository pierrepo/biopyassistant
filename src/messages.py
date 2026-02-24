"""Chatbot messages for out-of-scope, unrelated queries, and suggestions."""

MSGS_QUERY_NOT_RELATED = [
    (
        "Je suis désolé, je ne peux pas répondre à cette question. "
        "Mon domaine d'expertise est la programmation Python. "
        "N'hésite pas à me poser des questions liées à ce sujet,"
        "je serai ravi de t'aider."
    ),
    (
        "Désolé, je suis un assistant pour l'apprentissage de la programmation Python. "
        "Je ne suis pas en mesure de répondre à cette question."
    ),
    (
        "Je ne suis pas sûr de pouvoir répondre à cette question, car elle ne semble "
        "pas être liée à la programmation Python. Si tu as des questions sur Python, "
        "n'hésite pas à me les poser, je serai heureux de t'aider !"
    ),
]

MSGS_QUERY_OUT_OF_SCOPE_LEVEL = [
    (
        "Cette question fait référence à des notions qui ne sont pas encore abordées "
        "dans ce cours."
    ),
    (
        "Cette notion n'est pas encore abordée à votre niveau actuel "
        "et fait partie de la suite du programme."
    ),
    (
        "Cette question fait référence à des notions qui dépassent "
        "le cadre du niveau actuel de votre formation."
    ),
]

SUGGESTIONS = {
    ":green[:material/loop:] Comment écrire une boucle ?": (
        "Comment écrire une boucle ?"
    ),
    ":blue[:material/list_alt:] Quelle est la différence entre une liste et un set ?": (
        "Quelle est la différence entre une liste et un set ?"
    ),
    ":orange[:material/decimal_increase:] "
    "Comment afficher un float avec 2 chiffres avec la virgule ?": (
        "Comment afficher un float avec 2 chiffres avec la virgule ?"
    ),
}
