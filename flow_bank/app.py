"""
app.py — Flow Bank: منصة عربية لتقييم مخاطر الائتمان ببيانات حقيقية من UCI.

التشغيل:  streamlit run app.py
"""
import time

import numpy as np
import pandas as pd
import streamlit as st

import auth
import data_loader as dl
import models
import risk
import ui

st.set_page_config(page_title="Flow Bank", page_icon="🏦", layout="wide", initial_sidebar_state="collapsed")
ui.inject_css()

PAGES = ["الرئيسية", "تقييم فرد", "تقييم شركة", "أداء النماذج", "الخسائر المتوقعة"]


# ======================================================================= البيانات والنماذج
@st.cache_data(show_spinner=False)
def get_data():
    return dl.load_individuals(), dl.load_companies()


@st.cache_resource(show_spinner=False)
def get_models():
    ind, comp = get_data()
    return {"individuals": models.train_group(ind, "individuals"),
            "companies": models.train_group(comp, "companies")}


# ======================================================================= تسجيل الدخول
def load_auth_config():
    try:
        if "auth" in st.secrets:
            return dict(st.secrets["auth"])
    except Exception:  # لا يوجد ملف أسرار إطلاقًا
        pass
    return None


def login_page():
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        _login_box()


def _login_box():
    st.write("")
    st.markdown('<div class="fb-bar"><div class="fb-brand"><div class="fb-logo"></div><div>'
                '<div class="fb-name">Flow Bank</div><div class="fb-sub">تقييم مخاطر الائتمان</div></div></div></div>',
                unsafe_allow_html=True)
    cfg = load_auth_config()
    if not auth.valid_config(cfg):
        st.error("لم يُضبط حساب الدخول بعد. على جهازك: شغّلي python set_password.py ثم أعيدي التشغيل. "
                 "على Streamlit Cloud: الصقي قسم [auth] في Settings ← Secrets.")
        return
    wait = st.session_state.get("lock_until", 0) - time.time()
    if wait > 0:
        st.error(f"محاولات خاطئة كثيرة. أعيدي المحاولة بعد {int(wait) + 1} ثانية.")
        return
    with st.form("login"):
        st.markdown("#### تسجيل الدخول")
        username = st.text_input("اسم المستخدم", key="lg_user")
        password = st.text_input("كلمة المرور", type="password", key="lg_pass")
        ok = st.form_submit_button("دخول", type="primary", use_container_width=True)
    if ok:
        if auth.verify(username, password, cfg):
            st.session_state.update(auth=True, user=username.strip(), fails=0)
            st.rerun()
        fails = st.session_state.get("fails", 0) + 1
        st.session_state["fails"] = fails
        if fails >= 5:
            st.session_state.update(lock_until=time.time() + 60, fails=0)
            st.error("5 محاولات خاطئة. أُوقف الدخول لمدة دقيقة.")
        else:
            st.error(f"اسم المستخدم أو كلمة المرور غير صحيحة ({fails} من 5).")
    st.markdown('<p class="small" style="text-align:center">حساب واحد فقط، لمالكة المنصة.</p>', unsafe_allow_html=True)


# ======================================================================= الصفحات
def header():
    st.markdown(
        '<div class="fb-bar"><div class="fb-brand"><div class="fb-logo"></div><div>'
        '<div class="fb-name">Flow Bank</div><div class="fb-sub">تقييم مخاطر الائتمان ببيانات حقيقية</div></div></div>'
        f'<div class="fb-user">{st.session_state.get("user", "")}</div></div>', unsafe_allow_html=True)
    nav, out = st.columns([6, 1])
    with nav:
        page = st.radio("التنقل", PAGES, horizontal=True, key="page", label_visibility="collapsed")
    with out:
        if st.button("خروج", key="logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    return page


def page_home(ind, comp, res):
    st.markdown(
        '<div class="fb-hero"><h1>تقييم مخاطر الائتمان</h1>'
        '<p>تقيّم Flow Bank طلبات الأفراد والشركات بالخسارة المتوقعة EL = PD × LGD × EAD وجدول قرار واضح، '
        'وتعرض أداء نموذجين مدرَّبين على بيانات حقيقية من UCI Machine Learning Repository.</p></div>',
        unsafe_allow_html=True)
    st.write("")
    mi = models.metrics_table(res["individuals"], res["individuals"]["base_rate"])
    mc = models.metrics_table(res["companies"], res["companies"]["base_rate"])
    c = st.columns(4)
    c[0].markdown(ui.kpi("عملاء بطاقات الائتمان", f"{len(ind):,}".replace(",", " "), "تايوان، أبريل–سبتمبر 2005"), unsafe_allow_html=True)
    c[1].markdown(ui.kpi("نسبة التعثر الفعلية", ui.pct(ind[dl.CARD_TARGET].mean()), "تعثّر في الشهر التالي"), unsafe_allow_html=True)
    c[2].markdown(ui.kpi("شركات تايوانية", f"{len(comp):,}".replace(",", " "), "1999–2009"), unsafe_allow_html=True)
    c[3].markdown(ui.kpi("نسبة الإفلاس الفعلية", ui.pct(comp[dl.COMPANY_TARGET].mean()), "حسب قواعد بورصة تايوان"), unsafe_allow_html=True)

    st.subheader("ماذا تعني الألوان؟")
    st.markdown(
        f"{ui.tag('real')} قيمة موجودة كما هي في البيانات الأصلية. &nbsp; "
        f"{ui.tag('model')} احتمال حسبه نموذج تدرّب على البيانات. &nbsp; "
        f"{ui.tag('assume')} رقم نفترضه نحن للتوضيح، وتستطيعين تغييره. &nbsp; "
        f"{ui.tag('input')} قيمة أدخلتِها أنتِ.", unsafe_allow_html=True)

    st.subheader("ابدئي من هنا")
    a, b, c3 = st.columns(3)
    with a:
        st.markdown('<div class="fb-card"><h4>تقييم فرد</h4><p class="small">أدخلي بيانات العميل بنفسك: الدخل والمصاريف '
                    'والحد المطلوب والرصيد، واحصلي على الخسارة المتوقعة والقرار.</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="fb-card"><h4>تقييم شركة</h4><p class="small">أدخلي القوائم المالية للشركة ومبلغ القرض، '
                    'واحصلي على النسب المالية والخسارة المتوقعة والقرار.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="fb-card"><h4>أداء النماذج</h4><p class="small">قارني النموذجين بمقاييس ROC-AUC وRecall '
                    'وPrecision وBrier ومصفوفة الالتباس على بيانات لم يتدربا عليها.</p></div>', unsafe_allow_html=True)

    st.subheader("أفضل أداء على بيانات الاختبار")
    x, y = st.columns(2)
    for col, m, name in ((x, mi, "الأفراد"), (y, mc, "الشركات")):
        best = m.loc[m["ROC-AUC"].idxmax()]
        col.markdown(ui.kpi(f"{name}: {best['النموذج']}", f"{best['ROC-AUC']:.3f}",
                            "ROC-AUC (0.5 = تخمين عشوائي، 1 = تمييز كامل)"), unsafe_allow_html=True)

    st.subheader("مصادر البيانات وحدودها")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(
            '<div class="fb-card"><h4>الأفراد: Default of Credit Card Clients</h4>'
            '<p class="small">I-Cheng Yeh (2009)، UCI #350، رخصة CC BY 4.0. '
            '<a href="https://doi.org/10.24432/C55S3H" target="_blank">doi.org/10.24432/C55S3H</a></p>'
            '<ul class="small"><li>30 000 حامل بطاقة في تايوان. المبالغ الأصلية بالدولار التايواني، ونعرضها بالدينار '
            '(\u20661 NT$ = 4.135 DA\u2069، سعر أكتوبر 2026).</li>'
            '<li>التعثر هنا = عدم السداد في <b>الشهر التالي</b>، وليس خلال 12 شهرًا.</li>'
            '<li>لا نستعمل الجنس ولا الحالة الاجتماعية في التنبؤ، تجنّبًا للتمييز.</li></ul></div>',
            unsafe_allow_html=True)
    with s2:
        st.markdown(
            '<div class="fb-card"><h4>الشركات: Taiwanese Bankruptcy Prediction</h4>'
            '<p class="small">UCI #572 (2020)، مصدرها Taiwan Economic Journal، رخصة CC BY 4.0. '
            '<a href="https://doi.org/10.24432/C5004D" target="_blank">doi.org/10.24432/C5004D</a></p>'
            '<ul class="small"><li>6 819 شركة تايوانية (1999–2009)، 95 مؤشرًا ماليًا، منها 220 شركة أفلست.</li>'
            '<li>الحدث المتوقَّع هو <b>الإفلاس</b> حسب قواعد بورصة تايوان، والمصدر لا يحدد أفقًا زمنيًا للتنبؤ.</li>'
            '<li>أغلب المؤشرات مُقيَّسة من المصدر بين 0 و1، وبلا مبالغ قروض، لذلك EAD للشركات افتراض تدخلينه أنتِ.</li></ul></div>',
            unsafe_allow_html=True)


# ---------------------------------------------------------------------- نماذج دراسة الحالة
# الحقول مأخوذة من وثيقة «دراسات حالة: اتخاذ القرار الائتماني». النموذج فارغ: المستخدمة تُدخل كل القيم.
MARITAL = ["أعزب/عزباء", "متزوج/ة", "مطلق/ة", "أرمل/ة"]
HISTORY = ["لا يوجد", "جيد (بدون تأخر)", "تأخرات سابقة"]
GUARANTEE_IND = ["لا يوجد", "كفيل", "رهن أو ضمان عيني", "أخرى"]
GUARANTEE_COMP = ["لا يوجد", "معدات ومخزون", "عقار", "كفالة بنكية", "أخرى"]
DECISION_STYLE = {"accept": "low", "review": "mid", "reject": "high"}


def money_input(col, label, key, min_value=0.0):
    return col.number_input(f"{label} (DA)", min_value=min_value, value=None, step=1_000.0, format="%.0f",
                            key=key, placeholder="أدخلي المبلغ")


def missing_fields(values):
    return [name for name, v in values.items() if v is None or v == ""]


def decision_block(el, group, pd_, lgd, ead, ead_note):
    key, text = risk.decide(el, group)
    k = st.columns(4)
    k[0].markdown(ui.kpi(f"PD {ui.tag('input')}", ui.pct(pd_), "احتمال التعثر الذي أدخلتِه"), unsafe_allow_html=True)
    k[1].markdown(ui.kpi(f"LGD {ui.tag('assume')}", ui.pct(lgd, 0), "الخسارة عند التعثر"), unsafe_allow_html=True)
    k[2].markdown(ui.kpi("EAD", ui.money(ead, convert=False), ead_note), unsafe_allow_html=True)
    k[3].markdown(ui.kpi("EL", ui.money(el, convert=False), "الخسارة المتوقعة"), unsafe_allow_html=True)
    st.markdown(f'<div class="ecl-eq">EL = {ui.pct(pd_)} × {ui.pct(lgd, 0)} × {ui.money(ead, convert=False)} = '
                f'<b>{ui.money(el, convert=False)}</b></div>', unsafe_allow_html=True)
    st.markdown(f"#### القرار المقترح: {ui.pill(DECISION_STYLE[key], {DECISION_STYLE[key]: text})}", unsafe_allow_html=True)
    low, high = risk.DECISION_TABLES[group]["limits"]
    texts = risk.DECISION_TABLES[group]["texts"]
    da = lambda v: ui.money(v, convert=False)  # مبلغ معزول الاتجاه حتى لا ينقلب داخل النص العربي
    table = pd.DataFrame({"EL": [f"أقل من {da(low)}", f"من {da(low)} إلى {da(high)}", f"أكثر من {da(high)}"],
                          "القرار": [texts["accept"], texts["review"], texts["reject"]]})
    with st.expander("جدول اتخاذ القرار (من وثيقة دراسات الحالة)"):
        st.dataframe(table, hide_index=True, use_container_width=True)


def ratio_kpi(label, value, sub, as_pct=True):
    shown = "غير محسوبة" if value is None else (ui.pct(value) if as_pct else ui.fmt_num(value, 2))
    return ui.kpi(label, shown, sub)


def page_individual(ind):
    st.title("تقييم فرد")
    st.caption("أدخلي بيانات العميل كلها. لا يوجد أي حقل معبّأ مسبقًا. المبالغ بالدينار الجزائري.")
    st.markdown("### بيانات العميل")
    a, b, c = st.columns(3)
    name = a.text_input("الاسم", key="i_name", placeholder="اسم العميل")
    age = b.number_input("العمر (سنة)", min_value=18, max_value=100, value=None, step=1, key="i_age", placeholder="مثال: 32")
    marital = c.selectbox("الحالة الاجتماعية", MARITAL, index=None, key="i_marital", placeholder="اختاري")
    a, b, c = st.columns(3)
    job = a.text_input("المهنة", key="i_job", placeholder="مثال: موظف في شركة خاصة")
    years = b.number_input("مدة العمل (سنة)", min_value=0.0, max_value=60.0, value=None, step=0.5, key="i_years", placeholder="مثال: 3")
    history = c.selectbox("سجل ائتماني", HISTORY, index=None, key="i_hist", placeholder="اختاري")
    a, b, c = st.columns(3)
    income = money_input(a, "الدخل الشهري", "i_income")
    expenses = money_input(b, "المصاريف الشهرية", "i_exp")
    limit = money_input(c, "الحد الائتماني المطلوب", "i_limit")
    a, b, c = st.columns(3)
    balance = money_input(a, "الرصيد الحالي", "i_bal")
    guarantee = b.selectbox("الضمان", GUARANTEE_IND, index=None, key="i_guar", placeholder="اختاري")

    st.markdown("### معطيات الحساب")
    a, b, c = st.columns(3)
    pd_pct = a.number_input("PD: احتمال التعثر (%)", min_value=0.0, max_value=100.0, value=None, step=0.5, key="i_pd",
                            placeholder="مثال: 5")
    lgd = b.slider("LGD: نسبة الخسارة عند التعثر", 0.0, 1.0, risk.DEFAULT_LGD, 0.05, key="i_lgd")
    ccf = c.slider("CCF: نسبة استعمال الجزء غير المستعمل من الحد", 0.0, 1.0, 1.0, 0.05, key="i_ccf")
    st.caption(f"للمقارنة {ui.tag('real')}: في بيانات UCI تعثّر {ui.pct(ind[dl.CARD_TARGET].mean())} من حاملي البطاقات "
               "في الشهر التالي. القيمة الافتراضية \u2066CCF = 100%\u2069 افتراض محافظ: العميل يستعمل كامل الحد قبل التعثر.",
               unsafe_allow_html=True)

    required = {"الاسم": name.strip(), "العمر": age, "الحالة الاجتماعية": marital, "المهنة": job.strip(),
                "مدة العمل": years, "الدخل الشهري": income, "المصاريف الشهرية": expenses,
                "الحد الائتماني المطلوب": limit, "الرصيد الحالي": balance, "الضمان": guarantee,
                "سجل ائتماني": history, "PD": pd_pct}
    missing = missing_fields(required)
    if missing:
        st.info("أكملي الحقول لعرض النتيجة. الناقص: " + "، ".join(missing) + ".")
        return
    if limit <= 0:
        st.error("الحد الائتماني المطلوب يجب أن يكون أكبر من صفر.")
        return
    if income <= 0:
        st.error("الدخل الشهري يجب أن يكون أكبر من صفر لحساب المؤشرات.")
        return

    st.markdown(f"### النتيجة: {name.strip()}")
    pd_ = pd_pct / 100
    ead, drawn, undrawn = risk.card_ead(limit, balance, ccf)
    el = risk.ecl(pd_, lgd, ead)
    decision_block(el, "individuals", pd_, lgd, ead,
                   f"{ui.money(drawn, convert=False)} مستعمل + {ui.pct(ccf, 0)} × {ui.money(undrawn, convert=False)}")

    st.markdown("#### مؤشرات مساعدة من بياناتك")
    st.caption("هذه المؤشرات للتحليل فقط ولا تدخل في حساب EL. القرار يعتمد على EL وجدول القرار.")
    k = st.columns(4)
    k[0].markdown(ratio_kpi("المصاريف ÷ الدخل", risk.safe_ratio(expenses, income), "كم يذهب من الدخل للمصاريف"), unsafe_allow_html=True)
    k[1].markdown(ui.kpi("الدخل المتبقي شهريًا", ui.money(income - expenses, convert=False), "الدخل − المصاريف"), unsafe_allow_html=True)
    k[2].markdown(ratio_kpi("الحد ÷ الدخل السنوي", risk.safe_ratio(limit, income * 12), "حجم الحد مقارنة بالدخل"), unsafe_allow_html=True)
    k[3].markdown(ratio_kpi("الرصيد ÷ الحد", risk.safe_ratio(balance, limit), "نسبة الاستعمال الحالية"), unsafe_allow_html=True)
    notes = []
    if expenses > income:
        notes.append("المصاريف أكبر من الدخل.")
    if history == "تأخرات سابقة":
        notes.append("للعميل تأخرات سابقة في السداد.")
    if history == "لا يوجد":
        notes.append("لا يوجد سجل ائتماني: البنك لا يملك تاريخ سداد يحكم به.")
    if guarantee == "لا يوجد":
        notes.append("لا يوجد ضمان: الخسارة عند التعثر قد تكون أعلى.")
    if notes:
        st.warning(" ".join(notes))
    st.caption("الحالة الاجتماعية والمهنة تُسجَّلان للملف فقط ولا تؤثران في الحساب.")


def page_company(comp):
    st.title("تقييم شركة")
    st.caption("أدخلي بيانات الشركة كلها. لا يوجد أي حقل معبّأ مسبقًا. المبالغ بالدينار الجزائري.")
    st.markdown("### بيانات الشركة")
    a, b, c = st.columns(3)
    name = a.text_input("اسم الشركة", key="c_name", placeholder="مثال: شركة تصنيع صغيرة")
    loan = money_input(b, "مبلغ القرض المطلوب", "c_loan")
    term = c.number_input("المدة (شهر)", min_value=1, max_value=360, value=None, step=1, key="c_term", placeholder="مثال: 36")
    a, b, c = st.columns(3)
    assets = money_input(a, "إجمالي الأصول", "c_assets")
    debt = money_input(b, "إجمالي الدين", "c_debt")
    cur_assets = money_input(c, "الأصول المتداولة", "c_ca")
    a, b, c = st.columns(3)
    cur_liab = money_input(a, "الالتزامات المتداولة", "c_cl")
    profit = b.number_input("صافي الربح (DA)", value=None, step=1_000.0, format="%.0f", key="c_profit",
                            placeholder="سالب إذا كانت خسارة")
    ebit = c.number_input("EBIT: الربح قبل الفوائد والضرائب (DA)", value=None, step=1_000.0, format="%.0f", key="c_ebit",
                          placeholder="أدخلي المبلغ")
    a, b, _ = st.columns(3)
    interest = money_input(a, "الفوائد", "c_int")
    guarantee = b.selectbox("الضمان", GUARANTEE_COMP, index=None, key="c_guar", placeholder="اختاري")

    st.markdown("### معطيات الحساب")
    a, b = st.columns(2)
    pd_pct = a.number_input("PD: احتمال التعثر (%)", min_value=0.0, max_value=100.0, value=None, step=0.5, key="c_pd",
                            placeholder="مثال: 3")
    lgd = b.slider("LGD: نسبة الخسارة عند التعثر", 0.0, 1.0, risk.DEFAULT_LGD, 0.05, key="c_lgd")
    st.caption(f"للمقارنة {ui.tag('real')}: في بيانات UCI أفلست {ui.pct(comp[dl.COMPANY_TARGET].mean())} من الشركات التايوانية. "
               "\u2066EAD\u2069 = مبلغ القرض المطلوب كاملًا.", unsafe_allow_html=True)

    required = {"اسم الشركة": name.strip(), "مبلغ القرض المطلوب": loan, "المدة": term, "إجمالي الأصول": assets,
                "إجمالي الدين": debt, "الأصول المتداولة": cur_assets, "الالتزامات المتداولة": cur_liab,
                "صافي الربح": profit, "EBIT": ebit, "الفوائد": interest, "الضمان": guarantee, "PD": pd_pct}
    missing = missing_fields(required)
    if missing:
        st.info("أكملي الحقول لعرض النتيجة. الناقص: " + "، ".join(missing) + ".")
        return
    if loan <= 0:
        st.error("مبلغ القرض المطلوب يجب أن يكون أكبر من صفر.")
        return
    if cur_assets > assets:
        st.error("الأصول المتداولة لا يمكن أن تتجاوز إجمالي الأصول. راجعي الرقمين.")
        return

    st.markdown(f"### النتيجة: {name.strip()}")
    pd_ = pd_pct / 100
    el = risk.ecl(pd_, lgd, loan)
    decision_block(el, "companies", pd_, lgd, loan, f"مبلغ القرض على {int(term)} شهرًا")

    st.markdown("#### النسب المالية من بياناتك")
    st.caption("هذه النسب للتحليل فقط ولا تدخل في حساب EL. القرار يعتمد على EL وجدول القرار.")
    k = st.columns(3)
    k[0].markdown(ratio_kpi("نسبة المديونية", risk.safe_ratio(debt, assets), "إجمالي الدين ÷ إجمالي الأصول"), unsafe_allow_html=True)
    k[1].markdown(ratio_kpi("نسبة السيولة الجارية", risk.safe_ratio(cur_assets, cur_liab),
                            "الأصول المتداولة ÷ الالتزامات المتداولة", as_pct=False), unsafe_allow_html=True)
    k[2].markdown(ratio_kpi("تغطية الفوائد", risk.safe_ratio(ebit, interest), "EBIT ÷ الفوائد (عدد المرات)", as_pct=False),
                  unsafe_allow_html=True)
    k = st.columns(3)
    k[0].markdown(ratio_kpi("العائد على الأصول", risk.safe_ratio(profit, assets), "صافي الربح ÷ إجمالي الأصول"), unsafe_allow_html=True)
    k[1].markdown(ratio_kpi("القرض ÷ الأصول", risk.safe_ratio(loan, assets), "حجم القرض مقارنة بالأصول"), unsafe_allow_html=True)
    k[2].markdown(ui.kpi("القسط الشهري التقريبي", ui.money(loan / term, convert=False), "المبلغ ÷ المدة، بدون فوائد"),
                  unsafe_allow_html=True)
    notes = []
    cr = risk.safe_ratio(cur_assets, cur_liab)
    if cr is not None and cr < 1:
        notes.append("الأصول المتداولة أقل من الالتزامات المتداولة.")
    icr = risk.safe_ratio(ebit, interest)
    if icr is not None and icr < 1:
        notes.append("الربح قبل الفوائد لا يغطي الفوائد.")
    if profit < 0:
        notes.append("الشركة تحقق خسارة.")
    if guarantee == "لا يوجد":
        notes.append("لا يوجد ضمان: الخسارة عند التعثر قد تكون أعلى.")
    if notes:
        st.warning(" ".join(notes))


# ---------------------------------------------------------------------- أداء النماذج
def page_performance(res):
    st.title("أداء النماذج")
    group = st.radio("الفئة", ["individuals", "companies"], horizontal=True, key="perf_group",
                     format_func=lambda g: models.GROUPS[g]["name"])
    R = res[group]
    pos = "متعثر" if group == "individuals" else "مفلس"
    n_test, n_pos = len(R["y_test"]), int(R["y_test"].sum())
    st.caption(f"تدريب: {len(R['y_train']):,} سجل · اختبار: {n_test:,} سجل منها {n_pos:,} {pos}. "
               "الإعدادات اختيرت بالتحقق المتقاطع على بيانات التدريب فقط.")
    thr = st.slider("عتبة التصنيف: نعتبر السجل «" + pos + "» إذا تجاوز احتماله هذه القيمة",
                    0.01, 0.99, round(R["base_rate"], 2), 0.01, key=f"thr_{group}")
    st.caption(f"القيمة الافتراضية = نسبة الحالات «{pos}» في بيانات التدريب ({ui.pct(R['base_rate'])}). "
               "رفع العتبة يرفع Precision ويخفض Recall، والعكس. ROC-AUC وBrier لا يتأثران بالعتبة.")

    m = models.metrics_table(R, thr)
    show = m.drop(columns="kind").copy()
    st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns if c != "النموذج"}), hide_index=True,
                 use_container_width=True)
    best_auc = m.loc[m["ROC-AUC"].idxmax(), "النموذج"]
    best_brier = m.loc[m["Brier Score"].idxmin(), "النموذج"]
    st.markdown(f"- أفضل تمييز بين الحالات (ROC-AUC): **{best_auc}**.\n"
                f"- أدق احتمالات (أقل Brier): **{best_brier}**.\n"
                f"- «Brier (نموذج ساذج)» = خطأ من يعطي الجميع الاحتمال نفسه ({ui.pct(R['base_rate'])}). "
                "«تحسّن Brier» الموجب يعني أن النموذج أفضل من هذا التخمين.")

    st.subheader("مصفوفات الالتباس")
    c = st.columns(2)
    for col, kind in zip(c, ("logistic", "tree")):
        with col:
            st.markdown(f"**{models.MODEL_NAMES[kind]}** (العتبة {ui.pct(thr, 0)})")
            st.markdown(ui.confusion_html(models.confusion(R, kind, thr), pos=pos), unsafe_allow_html=True)
    st.caption(f"الخلايا الحمراء أخطاء: «فعلًا {pos} / توقّع سليم» خسارة مالية فاتت البنك، "
               "و«فعلًا سليم / توقّع " + pos + "» عميل جيد قد يُرفض.")

    a, b = st.columns(2)
    with a:
        st.subheader("منحنى ROC")
        ui.line_chart(models.roc_points(R), "fpr", "tpr", "model", "نسبة الإنذارات الخاطئة", "نسبة الحالات المكتشفة")
        st.caption("كلما اقترب المنحنى من الزاوية العليا كان التمييز أفضل؛ المتقطع = تخمين عشوائي.")
    with b:
        st.subheader("منحنى المعايرة")
        ui.line_chart(models.calibration_points(R), "predicted", "observed", "model", "الاحتمال المتوقَّع",
                      "النسبة الفعلية", points=True)
        st.caption("على الخط المتقطع = الاحتمال يطابق الواقع (مثلًا 20% متوقَّع ← 20% حدث فعلًا). هذا ما يقيسه Brier.")

    st.subheader("معنى المقاييس")
    st.markdown(
        f"- **ROC-AUC:** قدرة النموذج على وضع الحالات «{pos}» قبل السليمة في الترتيب. 0.5 = عشوائي، 1 = كامل.\n"
        f"- **Recall:** من بين كل {pos} فعلًا، كم اكتشف النموذج؟\n"
        f"- **Precision:** من بين من توقّع النموذج أنه {pos}، كم كان كذلك فعلًا؟\n"
        "- **Brier Score:** متوسط مربع الفرق بين الاحتمال والنتيجة (0 أو 1). كلما صغر كانت الاحتمالات أدق؛ "
        "مهم لأن ECL تستعمل الاحتمال نفسه.\n"
        "- **مصفوفة الالتباس:** عدد الإصابات والأخطاء بنوعيها عند العتبة المختارة.")
    with st.expander("إعدادات النموذجين المختارة"):
        for kind in ("logistic", "tree"):
            st.markdown(f"**{models.MODEL_NAMES[kind]}:** {R['best_params'][kind]} · "
                        f"ROC-AUC في التحقق المتقاطع {R['cv_auc'][kind]:.3f} · مدة التدريب {R['seconds'][kind]:.1f} ثانية")


# ---------------------------------------------------------------------- الخسائر المتوقعة
def page_ecl(ind, res):
    st.title("الخسائر المتوقعة (ECL)")
    settings_box()
    st.markdown(f"ECL = PD {ui.tag('model')} × LGD {ui.tag('assume')} × EAD {ui.tag('real')}{ui.tag('assume')}",
                unsafe_allow_html=True)
    t1, t2 = st.tabs(["محفظة الأفراد", "الشركات"])
    with t1:
        R = res["individuals"]
        s = st.columns(3)
        lgd = s[0].slider("LGD", 0.0, 1.0, risk.DEFAULT_LGD, 0.05, key="p_lgd")
        ccf = s[1].slider("CCF", 0.0, 1.0, risk.DEFAULT_CCF, 0.05, key="p_ccf")
        kind = s[2].selectbox("PD من", ["logistic", "tree"], format_func=models.MODEL_NAMES.get, key="p_kind")
        test = ind.set_index(dl.CARD_ID).loc[R["ids_test"]]
        pds = R["proba_test"][kind]
        ead, drawn, _ = risk.card_ead(test.LIMIT_BAL.to_numpy(), test.BILL_AMT1.to_numpy(), ccf)
        losses = pds * lgd * ead
        total_ecl, total_ead = float(losses.sum()), float(ead.sum())
        k = st.columns(4)
        k[0].markdown(ui.kpi("عدد العملاء", f"{len(test):,}".replace(",", " "), "مجموعة الاختبار"), unsafe_allow_html=True)
        k[1].markdown(ui.kpi("إجمالي EAD", ui.money(total_ead), "مستعمل + CCF × غير مستعمل"), unsafe_allow_html=True)
        k[2].markdown(ui.kpi("إجمالي ECL", ui.money(total_ecl), "لشهر واحد"), unsafe_allow_html=True)
        k[3].markdown(ui.kpi("ECL ÷ EAD", ui.pct(total_ecl / total_ead, 2), "معدل الخسارة المتوقعة"), unsafe_allow_html=True)
        st.caption(ui.RATE_NOTE)
        bands = st.session_state.get("bands_ind", risk.DEFAULT_BANDS["individuals"])
        lv = np.array([risk.risk_band(p, bands) for p in pds])
        by = pd.DataFrame({"المستوى": [risk.RISK_NAMES[x] for x in ("low", "mid", "high")],
                           "ECL": [ui.to_da(losses[lv == x].sum()) for x in ("low", "mid", "high")],
                           "العملاء": [(lv == x).sum() for x in ("low", "mid", "high")]})
        a, b = st.columns(2)
        with a:
            st.markdown("**ECL حسب مستوى المخاطر**")
            ui.bar_v(by, "المستوى", "ECL", "المستوى", "ECL (DA)")
        with b:
            st.markdown("**عدد العملاء حسب المستوى**")
            ui.bar_v(by, "المستوى", "العملاء", "المستوى", "عدد العملاء", color="#C98A0B")
        top = pd.DataFrame({"رقم العميل": test.index, "PD": pds, "EAD (DA)": ui.to_da(ead), "ECL (DA)": ui.to_da(losses)})
        top = top.sort_values("ECL (DA)", ascending=False).head(10)
        st.markdown("**أعلى 10 عملاء من حيث الخسارة المتوقعة**")
        st.dataframe(top.style.format({"PD": "{:.1%}", "EAD (DA)": "{:,.0f}", "ECL (DA)": "{:,.0f}"}),
                     hide_index=True, use_container_width=True)
        realized = float((ead * lgd)[R["y_test"].to_numpy() == 1].sum())
        st.caption(f"للمقارنة: لو طبّقنا الافتراضات نفسها على من تعثّروا فعلًا في الشهر التالي، تكون الخسارة "
                   f"{ui.money(realized)}. الفرق بينها وبين ECL يوضّح أثر جودة الاحتمالات (وأثر الافتراضات).")
    with t2:
        R = res["companies"]
        lgd_c = st.slider("LGD", 0.0, 1.0, risk.DEFAULT_LGD, 0.05, key="pc_lgd")
        kind_c = st.selectbox("PD من", ["logistic", "tree"], format_func=models.MODEL_NAMES.get, key="pc_kind")
        st.info("بيانات الشركات لا تحتوي مبالغ قروض، فلا يمكن حساب ECL بالمبالغ دون اختراع أرقام. "
                "نعرض بدلًا منها **معدل الخسارة المتوقعة لكل وحدة تعرّض** = PD × LGD. "
                "اضربيه في مبلغ القرض لتحصلي على ECL (أو استعملي صفحة «تقييم شركة»).")
        rate_ = R["proba_test"][kind_c] * lgd_c
        k = st.columns(3)
        k[0].markdown(ui.kpi("عدد الشركات", f"{len(rate_):,}".replace(",", " "), "مجموعة الاختبار"), unsafe_allow_html=True)
        k[1].markdown(ui.kpi("متوسط PD × LGD", ui.pct(rate_.mean(), 2), "لكل وحدة تعرّض"), unsafe_allow_html=True)
        k[2].markdown(ui.kpi("مثال: قرض 10 000 000 DA", ui.money(rate_.mean() * 10_000_000, "DA", convert=False), "ECL متوسطة"), unsafe_allow_html=True)


def settings_box():
    with st.expander("إعدادات حدود المخاطر (افتراض إداري)"):
        a, b = st.columns(2)
        st.session_state["bands_ind"] = a.slider("الأفراد: منخفض / مرتفع", 0.0, 1.0, risk.DEFAULT_BANDS["individuals"], 0.01, key="bi")
        st.session_state["bands_comp"] = b.slider("الشركات: منخفض / مرتفع", 0.0, 1.0, risk.DEFAULT_BANDS["companies"], 0.01, key="bc")
        for name, key in (("الأفراد", "bands_ind"), ("الشركات", "bands_comp")):
            lo, hi = st.session_state[key]
            if lo >= hi:
                st.warning(f"{name}: الحد الأول يجب أن يكون أقل من الثاني؛ أُعيدت القيم الافتراضية.")
                st.session_state[key] = risk.DEFAULT_BANDS["individuals" if key == "bands_ind" else "companies"]


# ======================================================================= التشغيل
def main():
    if not st.session_state.get("auth"):
        login_page()
        return
    page = header()
    try:
        with st.spinner("جارٍ تحميل البيانات الحقيقية وتدريب النموذجين (مرة واحدة، قد تستغرق دقيقة)..."):
            ind, comp = get_data()
            res = get_models()
    except dl.DataError as e:
        st.error(f"تعذّر تحميل البيانات: {e}")
        st.stop()
    try:
        if page == "الرئيسية":
            page_home(ind, comp, res)
        elif page == "تقييم فرد":
            page_individual(ind)
        elif page == "تقييم شركة":
            page_company(comp)
        elif page == "أداء النماذج":
            page_performance(res)
        else:
            page_ecl(ind, res)
    except ValueError as e:
        st.error(f"قيمة غير صالحة: {e}")
    st.markdown('<p class="small" style="margin-top:2rem">Flow Bank مشروع تعليمي. البيانات من UCI (رخصة CC BY 4.0). '
                'النتائج لا تصلح لقرارات إقراض حقيقية.</p>', unsafe_allow_html=True)


main()
