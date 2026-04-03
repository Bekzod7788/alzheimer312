def build_ai_text_summary(
    result: Dict[str, Any],
    prob_df: pd.DataFrame,
    borderline_msg: Optional[str],
    case_id: str,
    age: int,
    sex: str,
    notes: str,
) -> str:
    predicted = result["predicted_class"]
    confidence = result["confidence"]
    risk = get_risk_label(predicted)
    description = CLASS_DESCRIPTIONS.get(predicted, "Tasvir model bo‘yicha baholandi.")
    plan = RECOMMENDATION_MAP.get(predicted, {})

    top_rows = prob_df.head(3)
    top_summary = ", ".join(
        [f"{row['Class']} ({float(row['Probability']):.2%})" for _, row in top_rows.iterrows()]
    )

    summary: List[str] = []
    summary.append("### AI yordamida yozilgan kengaytirilgan sharh")
    summary.append(
        f"**Asosiy natija:** model tasvirni **{predicted}** sinfiga yaqin deb baholadi. "
        f"Hisoblangan ishonchlilik darajasi **{confidence:.2%}** bo‘ldi."
    )
    summary.append(f"**Klinik tavsif:** {description}")
    summary.append(f"**Risk talqini:** {risk}.")
    summary.append(f"**Top ehtimollar:** {top_summary}.")

    if plan.get("summary"):
        summary.append(f"**Ushbu klass bo‘yicha tushuntirish:** {plan['summary']}")

    if borderline_msg:
        summary.append(f"**Borderline ogohlantirish:** {borderline_msg}")

    if case_id or notes or sex != "Unknown":
        summary.append(
            f"**Kontekst:** Case ID — {case_id or 'N/A'}, yosh — {age}, jins — {sex}, foydalanuvchi izohi — {notes or 'yo‘q'}."
        )

    user_guidance = plan.get("user_guidance", [])
    if user_guidance:
        guidance_block = "\n".join([f"- {item}" for item in user_guidance])
        summary.append("### Foydalanuvchi yoki bemor uchun muhim izoh\n" + guidance_block)

    next_steps = plan.get("next_steps", [])
    if next_steps:
        next_block = "\n".join([f"- {item}" for item in next_steps])
        summary.append("### Keyingi to‘g‘ri yo‘nalish va tavsiya etiladigan qadamlar\n" + next_block)

    summary.append("### Etik va deontologik eslatma")
    summary.append("- Bu natija sun’iy intellekt asosidagi skrining xulosasi bo‘lib, yakuniy tibbiy tashxis hisoblanmaydi.")
    summary.append("- Natijani radiolog, nevrolog, psixiatr yoki boshqa malakali mutaxassis ko‘rigi bilan birga baholash lozim.")
    summary.append("- App foydalanuvchini qo‘rqitish yoki noto‘g‘ri xotirjam qilish uchun emas, balki to‘g‘ri klinik yo‘naltirish uchun ishlatiladi.")
    summary.append("- Agar foydalanuvchida tez kuchayib borayotgan xotira buzilishi, xulq o‘zgarishi yoki kundalik funksiyalar pasayishi bo‘lsa, tibbiy murojaatni kechiktirmaslik kerak.")

    return "\n\n".join(summary)