"""
Sample medical knowledge seeder for demo/testing.
Run: python utils/sample_data.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.vector_store import vector_store
from app.core.document_processor import document_processor

SAMPLE_DOCS = [
    {
        "text": """Diabetes mellitus is a group of metabolic diseases characterized by hyperglycemia
        resulting from defects in insulin secretion, insulin action, or both. The chronic hyperglycemia
        of diabetes is associated with long-term damage, dysfunction, and failure of different organs,
        especially the eyes, kidneys, nerves, heart, and blood vessels.
        Symptoms of diabetes include: excessive thirst (polydipsia), frequent urination (polyuria),
        unexplained weight loss, fatigue, blurred vision, slow-healing wounds, and frequent infections.
        Type 1 diabetes results from cellular-mediated autoimmune destruction of the beta cells of
        the pancreas. Type 2 diabetes results from a progressive loss of adequate beta-cell insulin
        secretion frequently on the background of insulin resistance.""",
        "metadata": {
            "source": "WHO Guidelines",
            "title": "WHO Global Report on Diabetes",
            "page": 12,
            "section": "Definition and Symptoms",
            "document_date": "2024-01-15",
            "guideline_version": "2024",
        }
    },
    {
        "text": """Hypertension Treatment Guidelines 2024:
        First-line treatment for hypertension includes lifestyle modifications:
        1. DASH diet (Dietary Approaches to Stop Hypertension)
        2. Sodium restriction to less than 2.3g/day
        3. Regular aerobic exercise (150 minutes/week moderate intensity)
        4. Weight loss if BMI > 25
        5. Limiting alcohol consumption
        6. Smoking cessation

        Pharmacological treatment is indicated when blood pressure remains above 140/90 mmHg
        despite lifestyle modifications. First-line medications include:
        - ACE inhibitors or ARBs (especially in diabetes or CKD)
        - Calcium channel blockers
        - Thiazide diuretics
        Target blood pressure: less than 130/80 mmHg for most adults.""",
        "metadata": {
            "source": "Clinical Research",
            "title": "JNC 8 Hypertension Guidelines Update",
            "page": 5,
            "section": "Treatment Recommendations",
            "document_date": "2024-03-01",
            "guideline_version": "JNC-8-2024",
        }
    },
    {
        "text": """COVID-19 Treatment Guidelines (2024 Update):
        Current WHO recommendations for COVID-19 management:

        Mild cases (no risk factors): Symptomatic treatment at home. Antipyretics for fever.
        Adequate hydration and rest. Monitoring for symptom progression.

        Moderate cases: Consider antiviral therapy (nirmatrelvir-ritonavir) within 5 days
        of symptom onset for high-risk patients. Supplemental oxygen if SpO2 < 94%.

        Severe/Critical cases: Dexamethasone 6mg/day for 10 days. Remdesivir for hospitalized
        patients requiring supplemental oxygen. IL-6 inhibitors (tocilizumab or baricitinib)
        for patients with severe inflammation.

        Vaccination remains the primary prevention strategy. Updated bivalent vaccines
        provide protection against Omicron subvariants.""",
        "metadata": {
            "source": "WHO Guidelines",
            "title": "WHO COVID-19 Clinical Management Guidelines",
            "page": 8,
            "section": "Treatment Protocols",
            "document_date": "2024-06-01",
            "guideline_version": "WHO-COVID-2024-v3",
        }
    },
    {
        "text": """Normal Blood Test Reference Ranges (Adults):

        Complete Blood Count (CBC):
        - Hemoglobin: Males 13.5-17.5 g/dL, Females 12.0-15.5 g/dL
        - White Blood Cells (WBC): 4,500-11,000 cells/mcL
        - Platelets: 150,000-400,000/mcL
        - Hematocrit: Males 41-53%, Females 36-46%

        Basic Metabolic Panel:
        - Fasting Blood Glucose: 70-100 mg/dL (normal), 100-125 mg/dL (prediabetes)
        - Creatinine: Males 0.74-1.35 mg/dL, Females 0.59-1.04 mg/dL
        - Sodium: 136-145 mEq/L
        - Potassium: 3.5-5.0 mEq/L
        - BUN: 7-20 mg/dL

        Lipid Panel:
        - Total Cholesterol: Less than 200 mg/dL (desirable)
        - LDL: Less than 100 mg/dL (optimal)
        - HDL: Men > 40 mg/dL, Women > 50 mg/dL
        - Triglycerides: Less than 150 mg/dL""",
        "metadata": {
            "source": "Medical Textbook",
            "title": "Harrison's Principles of Internal Medicine",
            "page": 245,
            "section": "Laboratory Reference Ranges",
            "document_date": "2023-09-01",
            "guideline_version": "21st Edition",
        }
    },
    {
        "text": """Myocardial Infarction (Heart Attack) - Recognition and Emergency Response:

        Warning Signs:
        - Chest pain, pressure, squeezing, or tightness (most common symptom)
        - Pain radiating to left arm, jaw, neck, or back
        - Shortness of breath
        - Nausea and vomiting
        - Cold sweats and lightheadedness
        - Women may experience atypical symptoms: fatigue, indigestion

        IMMEDIATE ACTION: Call emergency services (911) immediately. Do not drive yourself.
        Chew aspirin 325mg if not allergic and no contraindications.
        Time to treatment is critical: "Time is Muscle"

        STEMI Treatment: Primary percutaneous coronary intervention (PCI) is the gold standard
        if available within 90 minutes. Thrombolysis if PCI unavailable within 120 minutes.

        Long-term management: Dual antiplatelet therapy, statins, ACE inhibitors/ARBs,
        beta-blockers, and cardiac rehabilitation.""",
        "metadata": {
            "source": "Clinical Research",
            "title": "AHA/ACC STEMI Guidelines",
            "page": 3,
            "section": "Emergency Management",
            "document_date": "2023-12-15",
            "guideline_version": "AHA-2023",
        }
    },
]


def seed_sample_data():
    """Add sample medical documents to the vector store."""
    print("Seeding sample medical knowledge base...")
    total_chunks = 0

    for doc in SAMPLE_DOCS:
        chunks = document_processor.process_raw_text(
            doc["text"], doc["metadata"]
        )
        if chunks:
            vector_store.add_documents(chunks)
            total_chunks += len(chunks)
            print(f"  ✓ Added: {doc['metadata']['title']} ({len(chunks)} chunks)")

    stats = vector_store.get_collection_stats()
    print(f"\nKnowledge base ready: {stats['total_documents']} total chunks")
    print("Sample queries to try:")
    print("  - What are the symptoms of diabetes?")
    print("  - What are the latest COVID-19 treatment guidelines?")
    print("  - What is a normal blood glucose level?")
    print("  - How is hypertension treated?")


if __name__ == "__main__":
    seed_sample_data()