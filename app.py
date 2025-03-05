import streamlit as st
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
import pathlib
import uuid
from datetime import datetime

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


class Liability(BaseModel):
    liability_id: str = Field(description="the liability ID, starts with 'Z-'")
    liability_received_date: str = Field(
        description="date of receiving the liability, usually written after 'Zaprimljeno', in the format DD.MM.YYYY."
    )
    credit_issued_date: Optional[str] = Field(
        default=None,
        description="the date when the credit was issued, often the first date mentioned in the liability description, in the format DD.MM.YYYY.",
    )
    all_dates: list[str] = Field(
        default=[], description="all dates found in the liability description, in the format DD.MM.YYYY."
    )
    liability_amount: Optional[float] = Field(default=None, description="the monetary amount of the liability")
    creditor_info: Optional[str] = Field(
        default=None, description="the liability creditor, and their information if available (OIB, address, etc.)"
    )


class AllLiabilities(BaseModel):
    liabilities: list[Liability] = Field(description="a list of all liabilities")


TERETOVNICA_PROMPT = """You are a proficient document parser. Your task is to parse the given document written in Croatian.
The document lists all the liabilities of an individual, each under its own number ("Rbr." in document). Multiple liabilities can be found under the same "Rbr."
You have to parse every liability available, in order of their appearance.
You must extract the following information for each liability:
    - liability_id: the liability ID (starts with Z-)
    - liability_received_date: date when the liability was received (usually mentioned after "Zaprimljeno")
    - credit_issued_date: the date when the credit was issued, usually the first date in the liability description after the liabilities_received_date
    - all_dates: all dates found in the liability description
    - liability_amount: the monetary amount of the liability
    - creditor_info: creditor information (their name, OIB and address if available), the creditor is often a bank (but can be other entities, even people), the creditor is most likely written in capital letters at the end of the liability listing

Each liability entry is formatted as following:
| Rbr. | Sadržaj upisa | Iznos | Primjedba |
|------|---------------|-------|----------|
| <liability_number> | Zaprimljeno <liabilities_received_date> pod brojem <liabilities_id>\n<liabilities_description> | <liabilities_amount> | <optional_note> |

<liabilities_description> is a long text that contains the credit_start_date and creditor_info, among other information.
<credit_start_date> is the date when the credit was issued, usually the first date mentioned in the liability description.
<creditor_info> is the creditor information, often written at the end of the liability description.

Format your output as a JSON.

If there are no liabilities listed, return an empty list."""

prompt = ChatPromptTemplate.from_messages(
    [("system", TERETOVNICA_PROMPT), ("user", "Here is the document:\n{teretovnica}")]
)


def extract_data(uploaded_file):
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )
    rendered = converter(uploaded_file)
    text, _, images = text_from_rendered(rendered)
    teretovnica = text[text.index("# **C") :].replace("<br>", " ")
    return teretovnica


st.set_page_config(page_title="Liability Parser", page_icon="⚙️")
st.title("⚙️ Liability Parser ⚙️")
api_key = st.text_input("API key")
api_key_present = not (api_key is None or api_key == "")
if api_key:
    llm = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=api_key).with_structured_output(
        AllLiabilities
    )

    uploaded_file = st.file_uploader("Choose your .pdf file", type="pdf")
    if uploaded_file is not None:
        fp = pathlib.Path(str(uuid.uuid4()))
        try:
            fp.write_bytes(uploaded_file.getvalue())
            teretovnica = extract_data(str(fp))
        finally:
            if fp.is_file():
                fp.unlink()

        output = llm.invoke({"teretovnica": teretovnica})
        if len(output.liabilities) == 0:
            st.success("No liabilities found.")
        else:
            st.write("### Liabilities:")
            for liability in output.liabilities:
                if liability.liability_amount is not None:
                    if liability.credit_issued_date is None:
                        try:
                            parsed_dates = [datetime.strptime(date, "%d.%m.%Y.") for date in liability.all_dates]
                            liability.credit_issued_date = min(parsed_dates).strftime("%d.%m.%Y.")
                        except:
                            pass

                    with st.container(border=True):
                        st.write(f":blue[ID]: {liability.liability_id}")
                        st.write(f":blue[Amount]: {liability.liability_amount}")
                        st.write(f":blue[Received date]: {liability.liability_received_date}")
                        st.write(f":blue[Issued date]: {liability.credit_issued_date}")
                        st.write(f":blue[Creditor]: {liability.creditor_info}")
