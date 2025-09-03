from simple_agent_actions import navigate_with_agent
 
medical_document = """
Chief Complaint:
“Cough and shortness of breath on and off for a while.”

History of Present Illness:
Patient reports having had frequent respiratory infections for over 2 years. Episodes described as recurrent cough, wheezing, and chest tightness, usually worse with cold weather or when exposed to smoke/dust. States that symptoms have been ongoing for years, with gradual persistence of daily cough. Prior spirometry reportedly showed reduced FEV1 with partial reversibility. Findings consistent with airway obstruction noted over multiple visits. No recent acute flare-up reported. Currently denies fever or chills.

Past Medical History:
Known history of asthma since childhood. Patient recalls multiple episodes of “bronchitis” in the past, treated with antibiotics and inhalers.
No recent hospital admissions reported.

Medications:
Patient states using an inhaler “as needed.” Specific name not recalled. Reports occasional use of cough syrup in the past.

Allergies:
Denies drug allergies.

Family/Social History:
Father with history of asthma.
Non-smoker currently, but exposed to secondhand smoke at home.
Works indoors, reports dust exposure at workplace.

Review of Systems (limited):
Respiratory: Chronic cough, intermittent wheeze, shortness of breath on exertion.
General: No recent weight loss, no fever.
Cardiovascular: Denies chest pain.
Physical Exam (as observed):
Patient speaking in full sentences, no acute distress.
Mild wheeze noted on expiration.
No cyanosis.

Assessment:
Patient with history of asthma and long-standing episodes of recurrent respiratory infections likely obstructive in nature, currently presenting with persistent cough and wheezing. No signs of acute exacerbation at this time.
Plan (to be confirmed by attending):

Continue current inhaler use as prescribed.
Monitor for worsening symptoms (fever, increased shortness of breath).
Attending to evaluate for possible need of maintenance therapy.
Patient advised to avoid smoke exposure.
"""
 
print("="*80)
print("🏥 SIMPLE AGENT ACTIONS DEMO - ICD-10 Navigation")
print("="*80)
 
result = navigate_with_agent(medical_document, provider="cerebras", model="qwen-3-235b-a22b-instruct-2507")
 
print(f"\n" + "="*80)
print(f"📋 FINAL RESULTS")
print("="*80)
print(f"🎯 Final Code: {result['final_code']}")
print(f"📝 Final Diagnosis: {result['final_diagnosis']}")
print(f"📈 Navigation Steps: {result['total_steps']}")
print(f"🗺️  Complete Trajectory: {result.get('trajectory_summary', 'N/A')}")
print(f"✅ Navigation Complete: {result['completed']}")
 
if not result['completed']:
    print(f"\n⚠️  NAVIGATION STATUS:")
    print(f"   Navigation was halted before reaching a final diagnosis code.")
    print(f"   Review the step-by-step output above for specific halt reasons.")
 
if result['navigation_path']:
    print(f"\n📚 STEP-BY-STEP BREAKDOWN:")
    for step in result['navigation_path']:
        print(f"   Step {step['step']}: {step.get('hierarchy_level', 'Unknown')} → {step['selected_code']}")
        print(f"      Reason: {step['selection_reason']}")
 
print(f"\n💡 This demo shows the systematic navigation:")
print(f"   ROOT → Chapter → Block → Category → Subcategory → Final Code")
print(f"   Each step uses LLM reasoning to select the most appropriate path")