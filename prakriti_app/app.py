import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random

# ─────────────────────────────────────────────
# LANGUAGE DEFINITIONS
# ─────────────────────────────────────────────

LANGUAGES = {
    "English": "en",
    "हिंदी (Hindi)": "hi",
    "ಕನ್ನಡ (Kannada)": "kn"
}

UI_TEXT = {
    "en": {
        "tab1": "🌿 Prakriti Analysis",
        "tab2": "⚖️ Vikriti Analysis",
        "title1": "🌿 Prakriti Chatbot",
        "title2": "⚖️ Vikriti Analysis",
        "question_prefix": "How would you describe your",
        "choose_one": "Choose one:",
        "next": "Next Question ➔",
        "prev": "⬅ Previous Question",
        "complete": "Prakriti Analysis Complete!",
        "view_result": "View Result",
        "reset": "Reset",
        "your_prakriti": "### Your Prakriti:",
        "error_model": "Error loading Prakriti models:",
        "physical_symptoms": "🧍 Physical Symptoms",
        "mental_symptoms": "🧠 Mental Symptoms",
        "select_physical": "Select current physical issues:",
        "select_mental": "Select current mental issues:",
        "analyze": "Analyze My Vikriti",
        "aggravated_dosha": "### Current Aggravated Dosha:",
        "recommendations": "### 📋 Your Personalized Recommendations",
        "exercise_plan": "🧘 Exercise Plan",
        "meditation": "🧠 Meditation",
        "diet_plan": "🥗 Diet Plan",
        "music_therapy": "🎵 Music Therapy",
        "prediction_error": "Prediction Error:",
        "language_label": "🌐 Select Language",
        "recommended": "Recommended",
        "suggested": "Suggested",
    },
    "hi": {
        "tab1": "🌿 प्रकृति विश्लेषण",
        "tab2": "⚖️ विकृति विश्लेषण",
        "title1": "🌿 प्रकृति चैटबॉट",
        "title2": "⚖️ विकृति विश्लेषण",
        "question_prefix": "आप अपने",
        "question_suffix": "के बारे में क्या कहेंगे?",
        "choose_one": "एक विकल्प चुनें:",
        "next": "अगला प्रश्न ➔",
        "prev": "⬅ पिछला प्रश्न",
        "complete": "प्रकृति विश्लेषण पूर्ण!",
        "view_result": "परिणाम देखें",
        "reset": "रीसेट करें",
        "your_prakriti": "### आपकी प्रकृति:",
        "error_model": "प्रकृति मॉडल लोड करने में त्रुटि:",
        "physical_symptoms": "🧍 शारीरिक लक्षण",
        "mental_symptoms": "🧠 मानसिक लक्षण",
        "select_physical": "वर्तमान शारीरिक समस्याएं चुनें:",
        "select_mental": "वर्तमान मानसिक समस्याएं चुनें:",
        "analyze": "मेरी विकृति का विश्लेषण करें",
        "aggravated_dosha": "### वर्तमान प्रभावित दोष:",
        "recommendations": "### 📋 आपकी व्यक्तिगत सिफारिशें",
        "exercise_plan": "🧘 व्यायाम योजना",
        "meditation": "🧠 ध्यान",
        "diet_plan": "🥗 आहार योजना",
        "music_therapy": "🎵 संगीत चिकित्सा",
        "prediction_error": "भविष्यवाणी त्रुटि:",
        "language_label": "🌐 भाषा चुनें",
        "recommended": "अनुशंसित",
        "suggested": "सुझाया गया",
    },
    "kn": {
        "tab1": "🌿 ಪ್ರಕೃತಿ ವಿಶ್ಲೇಷಣೆ",
        "tab2": "⚖️ ವಿಕೃತಿ ವಿಶ್ಲೇಷಣೆ",
        "title1": "🌿 ಪ್ರಕೃತಿ ಚಾಟ್‌ಬಾಟ್",
        "title2": "⚖️ ವಿಕೃತಿ ವಿಶ್ಲೇಷಣೆ",
        "question_prefix": "ನಿಮ್ಮ",
        "question_suffix": "ಬಗ್ಗೆ ನೀವು ಹೇಗೆ ವಿವರಿಸುತ್ತೀರಿ?",
        "choose_one": "ಒಂದು ಆಯ್ಕೆ ಮಾಡಿ:",
        "next": "ಮುಂದಿನ ಪ್ರಶ್ನೆ ➔",
        "prev": "⬅ ಹಿಂದಿನ ಪ್ರಶ್ನೆ",
        "complete": "ಪ್ರಕೃತಿ ವಿಶ್ಲೇಷಣೆ ಪೂರ್ಣಗೊಂಡಿದೆ!",
        "view_result": "ಫಲಿತಾಂಶ ನೋಡಿ",
        "reset": "ಮರುಹೊಂದಿಸಿ",
        "your_prakriti": "### ನಿಮ್ಮ ಪ್ರಕೃತಿ:",
        "error_model": "ಪ್ರಕೃತಿ ಮಾದರಿಗಳನ್ನು ಲೋಡ್ ಮಾಡುವಲ್ಲಿ ದೋಷ:",
        "physical_symptoms": "🧍 ದೈಹಿಕ ಲಕ್ಷಣಗಳು",
        "mental_symptoms": "🧠 ಮಾನಸಿಕ ಲಕ್ಷಣಗಳು",
        "select_physical": "ಪ್ರಸ್ತುತ ದೈಹಿಕ ಸಮಸ್ಯೆಗಳನ್ನು ಆಯ್ಕೆ ಮಾಡಿ:",
        "select_mental": "ಪ್ರಸ್ತುತ ಮಾನಸಿಕ ಸಮಸ್ಯೆಗಳನ್ನು ಆಯ್ಕೆ ಮಾಡಿ:",
        "analyze": "ನನ್ನ ವಿಕೃತಿಯನ್ನು ವಿಶ್ಲೇಷಿಸಿ",
        "aggravated_dosha": "### ಪ್ರಸ್ತುತ ಉಲ್ಬಣಗೊಂಡ ದೋಷ:",
        "recommendations": "### 📋 ನಿಮ್ಮ ವೈಯಕ್ತಿಕ ಶಿಫಾರಸುಗಳು",
        "exercise_plan": "🧘 ವ್ಯಾಯಾಮ ಯೋಜನೆ",
        "meditation": "🧠 ಧ್ಯಾನ",
        "diet_plan": "🥗 ಆಹಾರ ಯೋಜನೆ",
        "music_therapy": "🎵 ಸಂಗೀತ ಚಿಕಿತ್ಸೆ",
        "prediction_error": "ಭವಿಷ್ಯವಾಣಿ ದೋಷ:",
        "language_label": "🌐 ಭಾಷೆ ಆಯ್ಕೆ ಮಾಡಿ",
        "recommended": "ಶಿಫಾರಸು ಮಾಡಲಾಗಿದೆ",
        "suggested": "ಸೂಚಿಸಲಾಗಿದೆ",
    }
}

# ─────────────────────────────────────────────
# PRAKRITI MAPPINGS — All 3 languages
# ─────────────────────────────────────────────

mappings = {
    "en": {
        'Body Size': {0: 'Large', 1: 'Medium', 2: 'Slim'},
        'Body Weight': {0: 'Heavy-You find it difficult to lose weight', 1: 'Light- You difficult to gain weight', 2: 'Moderate'},
        'Height': {0: 'Average-165 to 172 cm(for men) or 152 to 157 cm(for women)', 1: 'Short-below 165 cm(for men) or below 160 cm(for women)', 2: 'Tall- above 172 cm(for men) or above 157 cm(for women)'},
        'Bone Structure': {0: 'Heavy/Broad', 1: 'Light/Small', 2: 'Medium'},
        'Complexion': {0: 'Dark (Tans easily)', 1: 'Fair (Sunburns easily)', 2: 'White/Pale'},
        'General feel of skin': {0: 'Dry,Thin and Cool', 1: 'Smooth,Warm and Oily T-zone', 2: 'Thick,Moist and Cold'},
        'Texture of Skin': {0: 'Dry skin', 1: 'mixed skin(oily in some regions and dry in some regions)', 2: 'Oily skin'},
        'Hair Color': {0: 'Black', 1: 'Brown', 2: 'Red/Light Brown/Yellow'},
        'Appearance of Hair': {0: 'Dry/Brittle/Knotted', 1: 'Straight/Oily', 2: 'Thick/Curly'},
        'Shape of face': {0: 'Heart-shaped', 1: 'Round or square like', 2: 'Long or Angular'},
        'Eyes': {0: 'Big/Round', 1: 'Medium', 2: 'Small'},
        'Eyelashes': {0: 'Moderate', 1: 'Scanty', 2: 'Thick/Fused'},
        'Blinking of Eyes': {0: 'Excessive-more than 20 times per minute', 1: 'Low- 4 to 10 times per minute', 2: 'Stable- 12 to 20 times per minute'},
        'Cheeks': {0: 'Rounded/Plump', 1: 'Smooth/Flat', 2: 'Wrinkled/Sunken'},
        'Nose': {0: 'Narrow', 1: 'Pointed', 2: 'Rounded or large'},
        'Teeth and gums': {0: 'Big,Healthy', 1: 'Irregular teeth', 2: 'Medium'},
        'Lips': {0: 'Soft/Full', 1: 'Medium/Soft', 2: 'Thin/Dry'},
        'Nails': {0: 'Dry/Rough/Brittle', 1: 'Sharp/Flexible/Pink', 2: 'Thick/Oily/Smooth'},
        'Appetite': {0: 'Irregular- quantity/time changes', 1: 'Slow but steady', 2: 'Large appetite/fast eating'},
        'Liking tastes': {0: 'Tangy/Spicy/Bitter', 1: 'Sweet/Bitter', 2: 'Sweet/Sour/Salty'},
        'Metabolism Type': {0: 'Fast', 1: 'Moderate', 2: 'Slow'},
        'Climate Preference': {0: 'Winters', 1: 'Spring', 2: 'Summers'},
        'Stress Levels': {0: 'High', 1: 'Low', 2: 'Moderate'},
        'Sleep Patterns': {0: 'Long, undisturbed', 1: 'Moderate', 2: 'Short, frequently wake up'},
        'Dietary Habits': {0: 'Omnivorous', 1: 'Vegan', 2: 'Vegetarian'},
        'Physical Activity Level': {0: 'High', 1: 'Moderate', 2: 'Sedentary'},
        'Water Intake': {0: 'High (>4L)', 1: 'Low (<2L)', 2: 'Moderate (2-3L)'},
        'Digestion Quality': {0: 'Moderate', 1: 'Strong', 2: 'Weak'},
        'Skin Sensitivity': {0: 'Insensitive', 1: 'Normal', 2: 'Sensitive'}
    },
    "hi": {
        'शरीर का आकार': {0: 'बड़ा', 1: 'मध्यम', 2: 'पतला'},
        'शरीर का वजन': {0: 'भारी - वजन कम करना मुश्किल', 1: 'हल्का - वजन बढ़ाना मुश्किल', 2: 'सामान्य'},
        'ऊंचाई': {0: 'औसत - 165-172 सेमी (पुरुष) या 152-157 सेमी (महिला)', 1: 'छोटा - 165 सेमी से कम (पुरुष) या 160 सेमी से कम (महिला)', 2: 'लंबा - 172 सेमी से अधिक (पुरुष) या 157 सेमी से अधिक (महिला)'},
        'हड्डी की संरचना': {0: 'भारी/चौड़ी', 1: 'हल्की/छोटी', 2: 'मध्यम'},
        'रंग': {0: 'सांवला (आसानी से टैन होता है)', 1: 'गोरा (आसानी से जलता है)', 2: 'सफेद/पीला'},
        'त्वचा का सामान्य अहसास': {0: 'सूखी, पतली और ठंडी', 1: 'चिकनी, गर्म और टी-ज़ोन में तैलीय', 2: 'मोटी, नम और ठंडी'},
        'त्वचा की बनावट': {0: 'रूखी त्वचा', 1: 'मिश्रित त्वचा (कुछ क्षेत्रों में तैलीय, कुछ में रूखी)', 2: 'तैलीय त्वचा'},
        'बालों का रंग': {0: 'काला', 1: 'भूरा', 2: 'लाल/हल्का भूरा/पीला'},
        'बालों का रूप': {0: 'सूखे/भंगुर/उलझे', 1: 'सीधे/तैलीय', 2: 'घने/घुंघराले'},
        'चेहरे का आकार': {0: 'दिल के आकार का', 1: 'गोल या चौकोर', 2: 'लंबा या कोणीय'},
        'आंखें': {0: 'बड़ी/गोल', 1: 'मध्यम', 2: 'छोटी'},
        'पलकें': {0: 'सामान्य', 1: 'विरल', 2: 'घनी/जुड़ी हुई'},
        'आंखें झपकाना': {0: 'अत्यधिक - 20 से अधिक बार प्रति मिनट', 1: 'कम - 4 से 10 बार प्रति मिनट', 2: 'स्थिर - 12 से 20 बार प्रति मिनट'},
        'गाल': {0: 'गोल/भरे हुए', 1: 'चिकने/सपाट', 2: 'झुर्रीदार/धंसे हुए'},
        'नाक': {0: 'संकरी', 1: 'नुकीली', 2: 'गोल या बड़ी'},
        'दांत और मसूड़े': {0: 'बड़े, स्वस्थ', 1: 'अनियमित दांत', 2: 'मध्यम'},
        'होंठ': {0: 'मुलायम/भरे हुए', 1: 'मध्यम/मुलायम', 2: 'पतले/सूखे'},
        'नाखून': {0: 'सूखे/खुरदरे/भंगुर', 1: 'तीखे/लचीले/गुलाबी', 2: 'मोटे/तैलीय/चिकने'},
        'भूख': {0: 'अनियमित - मात्रा/समय बदलता रहता है', 1: 'धीमी लेकिन स्थिर', 2: 'अधिक भूख/तेज खाना'},
        'पसंदीदा स्वाद': {0: 'खट्टा/मसालेदार/कड़वा', 1: 'मीठा/कड़वा', 2: 'मीठा/खट्टा/नमकीन'},
        'चयापचय प्रकार': {0: 'तेज', 1: 'सामान्य', 2: 'धीमा'},
        'जलवायु प्राथमिकता': {0: 'सर्दी', 1: 'वसंत', 2: 'गर्मी'},
        'तनाव स्तर': {0: 'उच्च', 1: 'निम्न', 2: 'सामान्य'},
        'नींद का पैटर्न': {0: 'लंबी, अखंड नींद', 1: 'सामान्य', 2: 'कम, बार-बार जागना'},
        'आहार की आदतें': {0: 'सर्वाहारी', 1: 'शाकाहारी (वीगन)', 2: 'शाकाहारी'},
        'शारीरिक गतिविधि स्तर': {0: 'उच्च', 1: 'सामान्य', 2: 'निष्क्रिय'},
        'पानी का सेवन': {0: 'अधिक (>4 लीटर)', 1: 'कम (<2 लीटर)', 2: 'सामान्य (2-3 लीटर)'},
        'पाचन की गुणवत्ता': {0: 'सामान्य', 1: 'मजबूत', 2: 'कमजोर'},
        'त्वचा की संवेदनशीलता': {0: 'असंवेदनशील', 1: 'सामान्य', 2: 'संवेदनशील'}
    },
    "kn": {
        'ದೇಹದ ಗಾತ್ರ': {0: 'ದೊಡ್ಡದು', 1: 'ಮಧ್ಯಮ', 2: 'ತೆಳ್ಳಗೆ'},
        'ದೇಹದ ತೂಕ': {0: 'ಭಾರ - ತೂಕ ಕಳೆದುಕೊಳ್ಳಲು ಕಷ್ಟ', 1: 'ತಿಳಿ - ತೂಕ ಹೆಚ್ಚಿಸಲು ಕಷ್ಟ', 2: 'ಸಾಮಾನ್ಯ'},
        'ಎತ್ತರ': {0: 'ಸರಾಸರಿ - 165-172 ಸೆಮಿ (ಪುರುಷರು) ಅಥವಾ 152-157 ಸೆಮಿ (ಮಹಿಳೆಯರು)', 1: 'ಕುಳ್ಳ - 165 ಸೆಮಿಗಿಂತ ಕಡಿಮೆ (ಪುರುಷರು) ಅಥವಾ 160 ಸೆಮಿಗಿಂತ ಕಡಿಮೆ (ಮಹಿಳೆಯರು)', 2: 'ಎತ್ತರ - 172 ಸೆಮಿಗಿಂತ ಹೆಚ್ಚು (ಪುರುಷರು) ಅಥವಾ 157 ಸೆಮಿಗಿಂತ ಹೆಚ್ಚು (ಮಹಿಳೆಯರು)'},
        'ಮೂಳೆಯ ರಚನೆ': {0: 'ಭಾರ/ಅಗಲ', 1: 'ಹಗುರ/ಚಿಕ್ಕ', 2: 'ಮಧ್ಯಮ'},
        'ಚರ್ಮದ ಬಣ್ಣ': {0: 'ಕಪ್ಪು (ಸುಲಭವಾಗಿ ಟ್ಯಾನ್ ಆಗುತ್ತದೆ)', 1: 'ಬಿಳಿ (ಸುಲಭವಾಗಿ ಸುಡುತ್ತದೆ)', 2: 'ತಿಳಿ ಬಿಳಿ/ಪೇಲವ'},
        'ಚರ್ಮದ ಸಾಮಾನ್ಯ ಅನುಭವ': {0: 'ಒಣ, ತೆಳ್ಳಗೆ ಮತ್ತು ತಂಪು', 1: 'ನುಣ್ಣಗೆ, ಬೆಚ್ಚಗೆ ಮತ್ತು ಟಿ-ಝೋನ್‌ನಲ್ಲಿ ಎಣ್ಣೆ', 2: 'ದಪ್ಪ, ತೇವ ಮತ್ತು ತಂಪು'},
        'ಚರ್ಮದ ಗುಣ': {0: 'ಒಣ ಚರ್ಮ', 1: 'ಮಿಶ್ರ ಚರ್ಮ (ಕೆಲವು ಕಡೆ ಎಣ್ಣೆ, ಕೆಲವು ಕಡೆ ಒಣ)', 2: 'ಎಣ್ಣೆ ಚರ್ಮ'},
        'ಕೂದಲಿನ ಬಣ್ಣ': {0: 'ಕಪ್ಪು', 1: 'ಕಂದು', 2: 'ಕೆಂಪು/ತಿಳಿ ಕಂದು/ಹಳದಿ'},
        'ಕೂದಲಿನ ನೋಟ': {0: 'ಒಣ/ಸುಲಭವಾಗಿ ಮುರಿಯುವ/ಗಂಟು', 1: 'ನೇರ/ಎಣ್ಣೆಯ', 2: 'ದಪ್ಪ/ಸುರುಳಿ'},
        'ಮುಖದ ಆಕಾರ': {0: 'ಹೃದಯ ಆಕಾರ', 1: 'ದುಂಡು ಅಥವಾ ಚೌಕ', 2: 'ಉದ್ದ ಅಥವಾ ಕೋನೀಯ'},
        'ಕಣ್ಣುಗಳು': {0: 'ದೊಡ್ಡ/ದುಂಡು', 1: 'ಮಧ್ಯಮ', 2: 'ಚಿಕ್ಕ'},
        'ರೆಪ್ಪೆ': {0: 'ಸಾಮಾನ್ಯ', 1: 'ವಿರಳ', 2: 'ದಪ್ಪ/ಒಂದಾಗಿ'},
        'ಕಣ್ಣು ಮಿಟುಕಿಸುವಿಕೆ': {0: 'ಅತಿ ಹೆಚ್ಚು - ನಿಮಿಷಕ್ಕೆ 20ಕ್ಕಿಂತ ಹೆಚ್ಚು', 1: 'ಕಡಿಮೆ - ನಿಮಿಷಕ್ಕೆ 4 ರಿಂದ 10', 2: 'ಸ್ಥಿರ - ನಿಮಿಷಕ್ಕೆ 12 ರಿಂದ 20'},
        'ಕೆನ್ನೆ': {0: 'ದುಂಡು/ತುಂಬಿದ', 1: 'ನುಣ್ಣಗೆ/ಸಮತಟ್ಟು', 2: 'ನೆರಿಗೆ/ಕುಗ್ಗಿದ'},
        'ಮೂಗು': {0: 'ತೆಳ್ಳಗೆ', 1: 'ಚೂಪು', 2: 'ದುಂಡು ಅಥವಾ ದೊಡ್ಡ'},
        'ಹಲ್ಲು ಮತ್ತು ವಸಡು': {0: 'ದೊಡ್ಡ, ಆರೋಗ್ಯಕರ', 1: 'ಅನಿಯಮಿತ ಹಲ್ಲುಗಳು', 2: 'ಮಧ್ಯಮ'},
        'ತುಟಿ': {0: 'ಮೃದು/ತುಂಬಿದ', 1: 'ಮಧ್ಯಮ/ಮೃದು', 2: 'ತೆಳ್ಳಗೆ/ಒಣ'},
        'ಉಗುರುಗಳು': {0: 'ಒಣ/ಒರಟು/ಸುಲಭ ಮುರಿಯುವ', 1: 'ಚೂಪು/ಹೊಂದಿಕೊಳ್ಳುವ/ಗುಲಾಬಿ', 2: 'ದಪ್ಪ/ಎಣ್ಣೆ/ನುಣ್ಣಗೆ'},
        'ಹಸಿವು': {0: 'ಅನಿಯಮಿತ - ಪ್ರಮಾಣ/ಸಮಯ ಬದಲಾಗುತ್ತದೆ', 1: 'ನಿಧಾನ ಆದರೆ ಸ್ಥಿರ', 2: 'ಹೆಚ್ಚು ಹಸಿವು/ವೇಗವಾಗಿ ತಿನ್ನುವ'},
        'ರುಚಿ ಆದ್ಯತೆ': {0: 'ಖಾರ/ಖಾರ/ಕಹಿ', 1: 'ಸಿಹಿ/ಕಹಿ', 2: 'ಸಿಹಿ/ಹುಳಿ/ಉಪ್ಪು'},
        'ಚಯಾಪಚಯ ವಿಧ': {0: 'ವೇಗ', 1: 'ಮಧ್ಯಮ', 2: 'ನಿಧಾನ'},
        'ಹವಾಮಾನ ಆದ್ಯತೆ': {0: 'ಚಳಿಗಾಲ', 1: 'ವಸಂತ', 2: 'ಬೇಸಿಗೆ'},
        'ಒತ್ತಡದ ಮಟ್ಟ': {0: 'ಹೆಚ್ಚು', 1: 'ಕಡಿಮೆ', 2: 'ಮಧ್ಯಮ'},
        'ನಿದ್ರೆಯ ಮಾದರಿ': {0: 'ದೀರ್ಘ, ತೊಂದರೆಯಿಲ್ಲದ', 1: 'ಮಧ್ಯಮ', 2: 'ಕಡಿಮೆ, ಆಗಾಗ ಎದ್ದೇಳುವ'},
        'ಆಹಾರ ಪದ್ಧತಿ': {0: 'ಸರ್ವಭಕ್ಷಕ', 1: 'ಸಂಪೂರ್ಣ ಸಸ್ಯಾಹಾರಿ', 2: 'ಸಸ್ಯಾಹಾರಿ'},
        'ದೈಹಿಕ ಚಟುವಟಿಕೆಯ ಮಟ್ಟ': {0: 'ಹೆಚ್ಚು', 1: 'ಮಧ್ಯಮ', 2: 'ನಿಷ್ಕ್ರಿಯ'},
        'ನೀರು ಸೇವನೆ': {0: 'ಹೆಚ್ಚು (>4 ಲೀ)', 1: 'ಕಡಿಮೆ (<2 ಲೀ)', 2: 'ಮಧ್ಯಮ (2-3 ಲೀ)'},
        'ಜೀರ್ಣಕ್ರಿಯೆ ಗುಣಮಟ್ಟ': {0: 'ಮಧ್ಯಮ', 1: 'ಪ್ರಬಲ', 2: 'ದುರ್ಬಲ'},
        'ಚರ್ಮದ ಸಂವೇದನಶೀಲತೆ': {0: 'ಸಂವೇದನರಹಿತ', 1: 'ಸಾಮಾನ್ಯ', 2: 'ಸಂವೇದನಶೀಲ'}
    }
}

# ─────────────────────────────────────────────
# VIKRITI SYMPTOMS — All 3 languages
# ─────────────────────────────────────────────

physical_symptoms_lang = {
    "en": [
        "Dry skin", "Constipation", "Irregular appetite", "Cold intolerance", "Fatigue",
        "Acidity", "Burning sensation on the skin", "Excessive sweating", "Headache", "Loose stools",
        "Excessive Hunger", "Inflammation", "Lethargy", "Weight gain", "Slow metabolism",
        "Excess sleep", "Excessive mucus", "Body feels heavy"
    ],
    "hi": [
        "रूखी त्वचा", "कब्ज", "अनियमित भूख", "ठंड असहिष्णुता", "थकान",
        "एसिडिटी", "त्वचा पर जलन", "अत्यधिक पसीना", "सिरदर्द", "ढीला मल",
        "अत्यधिक भूख", "सूजन", "सुस्ती", "वजन बढ़ना", "धीमा चयापचय",
        "अत्यधिक नींद", "अत्यधिक बलगम", "शरीर भारी लगना"
    ],
    "kn": [
        "ಒಣ ಚರ್ಮ", "ಮಲಬದ್ಧತೆ", "ಅನಿಯಮಿತ ಹಸಿವು", "ಚಳಿ ಅಸಹಿಷ್ಣುತೆ", "ಆಯಾಸ",
        "ಆಮ್ಲೀಯತೆ", "ಚರ್ಮದ ಮೇಲೆ ಉರಿ", "ಅತಿಯಾದ ಬೆವರು", "ತಲೆನೋವು", "ಸಡಿಲ ಮಲ",
        "ಅತಿಯಾದ ಹಸಿವು", "ಉರಿಯೂತ", "ಸೋಮಾರಿತನ", "ತೂಕ ಹೆಚ್ಚಾಗುವಿಕೆ", "ನಿಧಾನ ಚಯಾಪಚಯ",
        "ಅತಿಯಾದ ನಿದ್ರೆ", "ಅತಿಯಾದ ಲೋಳೆ", "ದೇಹ ಭಾರವಾಗಿದೆ"
    ]
}

mental_symptoms_lang = {
    "en": ["Anxiety", "Insomnia", "Restlessness", "Irritability", "Depression", "Motivation loss"],
    "hi": ["चिंता", "अनिद्रा", "बेचैनी", "चिड़चिड़ापन", "अवसाद", "प्रेरणा की कमी"],
    "kn": ["ಆತಂಕ", "ನಿದ್ರಾಹೀನತೆ", "ಚಡಪಡಿಕೆ", "ಕಿರಿಕಿರಿ", "ಖಿನ್ನತೆ", "ಪ್ರೇರಣೆ ಕೊರತೆ"]
}

# Mapping from localized symptom display name → internal column key
rename_map_lang = {
    "en": {
        "Dry skin": "Dry_skin", "Constipation": "Constipation", "Irregular appetite": "Irregular_appetite",
        "Cold intolerance": "Cold_intolerance", "Fatigue": "Fatigue", "Acidity": "Acidity",
        "Burning sensation on the skin": "Burning_sensation", "Excessive sweating": "Sweating",
        "Headache": "Headache", "Loose stools": "Loose_stools", "Excessive Hunger": "Hunger",
        "Inflammation": "Inflammation", "Lethargy": "Lethargy", "Weight gain": "Weight_gain",
        "Slow metabolism": "Slow_metabolism", "Excess sleep": "Excess_sleep", "Excessive mucus": "Mucus",
        "Body feels heavy": "Heavy_body", "Anxiety": "Anxiety", "Insomnia": "Insomnia",
        "Restlessness": "Restlessness", "Irritability": "Irritability", "Depression": "Depression",
        "Motivation loss": "Motivation_loss"
    },
    "hi": {
        "रूखी त्वचा": "Dry_skin", "कब्ज": "Constipation", "अनियमित भूख": "Irregular_appetite",
        "ठंड असहिष्णुता": "Cold_intolerance", "थकान": "Fatigue", "एसिडिटी": "Acidity",
        "त्वचा पर जलन": "Burning_sensation", "अत्यधिक पसीना": "Sweating",
        "सिरदर्द": "Headache", "ढीला मल": "Loose_stools", "अत्यधिक भूख": "Hunger",
        "सूजन": "Inflammation", "सुस्ती": "Lethargy", "वजन बढ़ना": "Weight_gain",
        "धीमा चयापचय": "Slow_metabolism", "अत्यधिक नींद": "Excess_sleep", "अत्यधिक बलगम": "Mucus",
        "शरीर भारी लगना": "Heavy_body", "चिंता": "Anxiety", "अनिद्रा": "Insomnia",
        "बेचैनी": "Restlessness", "चिड़चिड़ापन": "Irritability", "अवसाद": "Depression",
        "प्रेरणा की कमी": "Motivation_loss"
    },
    "kn": {
        "ಒಣ ಚರ್ಮ": "Dry_skin", "ಮಲಬದ್ಧತೆ": "Constipation", "ಅನಿಯಮಿತ ಹಸಿವು": "Irregular_appetite",
        "ಚಳಿ ಅಸಹಿಷ್ಣುತೆ": "Cold_intolerance", "ಆಯಾಸ": "Fatigue", "ಆಮ್ಲೀಯತೆ": "Acidity",
        "ಚರ್ಮದ ಮೇಲೆ ಉರಿ": "Burning_sensation", "ಅತಿಯಾದ ಬೆವರು": "Sweating",
        "ತಲೆನೋವು": "Headache", "ಸಡಿಲ ಮಲ": "Loose_stools", "ಅತಿಯಾದ ಹಸಿವು": "Hunger",
        "ಉರಿಯೂತ": "Inflammation", "ಸೋಮಾರಿತನ": "Lethargy", "ತೂಕ ಹೆಚ್ಚಾಗುವಿಕೆ": "Weight_gain",
        "ನಿಧಾನ ಚಯಾಪಚಯ": "Slow_metabolism", "ಅತಿಯಾದ ನಿದ್ರೆ": "Excess_sleep", "ಅತಿಯಾದ ಲೋಳೆ": "Mucus",
        "ದೇಹ ಭಾರವಾಗಿದೆ": "Heavy_body", "ಆತಂಕ": "Anxiety", "ನಿದ್ರಾಹೀನತೆ": "Insomnia",
        "ಚಡಪಡಿಕೆ": "Restlessness", "ಕಿರಿಕಿರಿ": "Irritability", "ಖಿನ್ನತೆ": "Depression",
        "ಪ್ರೇರಣೆ ಕೊರತೆ": "Motivation_loss"
    }
}

# Prakriti result labels
p_lookup_lang = {
    "en": {0: 'Kapha', 1: 'Pitta', 2: 'Vata', 3: 'Pitta + Kapha', 4: 'Vata + Kapha', 5: 'Vata + Pitta'},
    "hi": {0: 'कफ', 1: 'पित्त', 2: 'वात', 3: 'पित्त + कफ', 4: 'वात + कफ', 5: 'वात + पित्त'},
    "kn": {0: 'ಕಫ', 1: 'ಪಿತ್ತ', 2: 'ವಾತ', 3: 'ಪಿತ್ತ + ಕಫ', 4: 'ವಾತ + ಕಫ', 5: 'ವಾತ + ಪಿತ್ತ'}
}

# ─────────────────────────────────────────────
# PLAN TEMPLATES — All 3 languages
# ─────────────────────────────────────────────

plan_templates_lang = {
    "en": {
        "Vata": {
            "exercise": ["Balasana", "Slow walking", "Pashchimottanasana", "Veerabhadrasana", "Vrikshasana", "Pranayama"],
            "meditation": "Deep breathing and guided grounding meditation.",
            "diet": ["Warm cooked meals", "Carrots", "Sweet potatoes", "Soaked nuts", "Khichadi with Ghee", "Warm Soups"],
            "music": ["Kalyani", "Neelambari", "Pantuvarali", "Anandabhairavi", "Sahana"]
        },
        "Pitta": {
            "exercise": ["Light jogging", "Swimming", "Chandra Namaskara", "Brisk Walking"],
            "meditation": "Cooling breathing (Sheetali) and mindfulness.",
            "diet": ["Coconut water", "Watermelons", "Cucumbers", "Mint", "Coriander", "Fresh Fruits"],
            "music": ["Hamsadhwani", "Mohanam", "Kapi", "Madhyamavati", "Yamuna Kalyani"]
        },
        "Kapha": {
            "exercise": ["Running", "High-intensity Cardio", "Surya Namaskara"],
            "meditation": "Active meditation and Bhastrika Pranayama.",
            "diet": ["Light spicy food", "Millets", "Warm meals", "Ginger tea", "Buckwheat", "Leafy vegetables"],
            "music": ["Bhairavi", "Revati", "Kharaharapriya"]
        }
    },
    "hi": {
        "Vata": {
            "exercise": ["बालासन", "धीमी चलना", "पश्चिमोत्तानासन", "वीरभद्रासन", "वृक्षासन", "प्राणायाम"],
            "meditation": "गहरी सांस लेना और निर्देशित ग्राउंडिंग ध्यान।",
            "diet": ["गर्म पका हुआ भोजन", "गाजर", "शकरकंद", "भीगे हुए मेवे", "घी के साथ खिचड़ी", "गर्म सूप"],
            "music": ["कल्याणी", "नीलांबरी", "पंतुवरली", "आनंदभैरवी", "सहाना"]
        },
        "Pitta": {
            "exercise": ["हल्की जॉगिंग", "तैराकी", "चंद्र नमस्कार", "तेज चलना"],
            "meditation": "शीतलन श्वास (शीतली) और माइंडफुलनेस।",
            "diet": ["नारियल पानी", "तरबूज", "खीरा", "पुदीना", "धनिया", "ताजे फल"],
            "music": ["हंसध्वनि", "मोहनम", "कापी", "मध्यमावती", "यमुना कल्याणी"]
        },
        "Kapha": {
            "exercise": ["दौड़ना", "उच्च तीव्रता कार्डियो", "सूर्य नमस्कार"],
            "meditation": "सक्रिय ध्यान और भस्त्रिका प्राणायाम।",
            "diet": ["हल्का मसालेदार भोजन", "बाजरा", "गर्म भोजन", "अदरक की चाय", "कुट्टू", "पत्तेदार सब्जियां"],
            "music": ["भैरवी", "रेवती", "खरहरप्रिया"]
        }
    },
    "kn": {
        "Vata": {
            "exercise": ["ಬಾಲಾಸನ", "ನಿಧಾನ ನಡಿಗೆ", "ಪಶ್ಚಿಮೋತ್ತಾನಾಸನ", "ವೀರಭದ್ರಾಸನ", "ವೃಕ್ಷಾಸನ", "ಪ್ರಾಣಾಯಾಮ"],
            "meditation": "ಆಳವಾದ ಉಸಿರಾಟ ಮತ್ತು ಮಾರ್ಗದರ್ಶಿ ಧ್ಯಾನ.",
            "diet": ["ಬಿಸಿ ಬೇಯಿಸಿದ ಊಟ", "ಕ್ಯಾರೆಟ್", "ಸಿಹಿ ಆಲೂ", "ನೆನೆಸಿದ ಬೀಜಗಳು", "ತುಪ್ಪ ಸೇರಿಸಿ ಖಿಚಡಿ", "ಬಿಸಿ ಸೂಪ್"],
            "music": ["ಕಲ್ಯಾಣಿ", "ನೀಲಾಂಬರಿ", "ಪಂತುವರಳಿ", "ಆನಂದಭೈರವಿ", "ಸಹಾನ"]
        },
        "Pitta": {
            "exercise": ["ಲಘು ಜಾಗಿಂಗ್", "ಈಜು", "ಚಂದ್ರ ನಮಸ್ಕಾರ", "ವೇಗದ ನಡಿಗೆ"],
            "meditation": "ತಂಪಾಗಿಸುವ ಉಸಿರಾಟ (ಶೀತಲಿ) ಮತ್ತು ಮೈಂಡ್‌ಫುಲ್‌ನೆಸ್.",
            "diet": ["ತೆಂಗಿನ ನೀರು", "ಕಲ್ಲಂಗಡಿ", "ಸೌತೆಕಾಯಿ", "ಪುದೀನ", "ಕೊತ್ತಂಬರಿ", "ತಾಜಾ ಹಣ್ಣುಗಳು"],
            "music": ["ಹಂಸಧ್ವನಿ", "ಮೋಹನಮ್", "ಕಾಪಿ", "ಮಧ್ಯಮಾವತಿ", "ಯಮುನಾ ಕಲ್ಯಾಣಿ"]
        },
        "Kapha": {
            "exercise": ["ಓಟ", "ಹೆಚ್ಚು ತೀವ್ರತೆಯ ಕಾರ್ಡಿಯೋ", "ಸೂರ್ಯ ನಮಸ್ಕಾರ"],
            "meditation": "ಸಕ್ರಿಯ ಧ್ಯಾನ ಮತ್ತು ಭಸ್ತ್ರಿಕಾ ಪ್ರಾಣಾಯಾಮ.",
            "diet": ["ಲಘು ಖಾರ ಆಹಾರ", "ರಾಗಿ/ಜೋಳ", "ಬಿಸಿ ಊಟ", "ಶುಂಠಿ ಚಹಾ", "ಕ್ಯಾಕ್ಟಸ್ ಧಾನ್ಯ", "ಎಲೆ ತರಕಾರಿ"],
            "music": ["ಭೈರವಿ", "ರೇವತಿ", "ಖರಹರಪ್ರಿಯ"]
        }
    }
}

raga_data_lang = {
    "en": {
        "Kalyani": {"benefit": "to boost energy and remove fear", "symptoms": ["Fatigue", "Anxiety", "Motivation loss"]},
        "Neelambari": {"benefit": "to treat insomnia and bring about calm sleep", "symptoms": ["Insomnia", "Restlessness"]},
        "Pantuvarali": {"benefit": "to alleviate mental dilemmas and sadness", "symptoms": ["Depression", "Motivation loss", "Anxiety"]},
        "Anandabhairavi": {"benefit": "to suppress stomach pain and manage hypertension", "symptoms": ["Acidity", "Inflammation"]},
        "Sahana": {"benefit": "to control high blood pressure and calm the mind", "symptoms": ["Irritability", "Anxiety"]},
        "Hamsadhwani": {"benefit": "to reduce tension and promote positive thinking", "symptoms": ["Restlessness", "Anxiety", "Irritability"]},
        "Mohanam": {"benefit": "to manage stress, headaches, and indigestion", "symptoms": ["Headache", "Irregular appetite", "Slow metabolism"]},
        "Kapi": {"benefit": "to reduce depression and nervous tension", "symptoms": ["Depression", "Anxiety", "Insomnia"]},
        "Madhyamavati": {"benefit": "to treat nervous complaints and body pain", "symptoms": ["Fatigue", "Body feels heavy"]},
        "Yamuna Kalyani": {"benefit": "for mental rejuvenation", "symptoms": ["Lethargy", "Fatigue", "Motivation loss"]},
        "Bhairavi": {"benefit": "to help with chronic headaches and inflammation", "symptoms": ["Headache", "Inflammation", "Excessive mucus"]},
        "Revati": {"benefit": "to calm the senses and reduce anxiety", "symptoms": ["Anxiety", "Restlessness", "Insomnia"]},
        "Kharaharapriya": {"benefit": "to help with nervous irritation", "symptoms": ["Irritability", "Restlessness"]}
    },
    "hi": {
        "Kalyani": {"benefit": "ऊर्जा बढ़ाने और भय दूर करने के लिए", "symptoms": ["थकान", "चिंता", "प्रेरणा की कमी"]},
        "Neelambari": {"benefit": "अनिद्रा का उपचार और शांत नींद लाने के लिए", "symptoms": ["अनिद्रा", "बेचैनी"]},
        "Pantuvarali": {"benefit": "मानसिक दुविधा और उदासी दूर करने के लिए", "symptoms": ["अवसाद", "प्रेरणा की कमी", "चिंता"]},
        "Anandabhairavi": {"benefit": "पेट दर्द और उच्च रक्तचाप नियंत्रण के लिए", "symptoms": ["एसिडिटी", "सूजन"]},
        "Sahana": {"benefit": "उच्च रक्तचाप नियंत्रण और मन शांत करने के लिए", "symptoms": ["चिड़चिड़ापन", "चिंता"]},
        "Hamsadhwani": {"benefit": "तनाव कम करने और सकारात्मक सोच बढ़ाने के लिए", "symptoms": ["बेचैनी", "चिंता", "चिड़चिड़ापन"]},
        "Mohanam": {"benefit": "तनाव, सिरदर्द और अपच प्रबंधन के लिए", "symptoms": ["सिरदर्द", "अनियमित भूख", "धीमा चयापचय"]},
        "Kapi": {"benefit": "अवसाद और नर्वस तनाव कम करने के लिए", "symptoms": ["अवसाद", "चिंता", "अनिद्रा"]},
        "Madhyamavati": {"benefit": "नर्वस समस्याओं और शरीर दर्द के लिए", "symptoms": ["थकान", "शरीर भारी लगना"]},
        "Yamuna Kalyani": {"benefit": "मानसिक ताजगी के लिए", "symptoms": ["सुस्ती", "थकान", "प्रेरणा की कमी"]},
        "Bhairavi": {"benefit": "पुराने सिरदर्द और सूजन में सहायता के लिए", "symptoms": ["सिरदर्द", "सूजन", "अत्यधिक बलगम"]},
        "Revati": {"benefit": "इंद्रियों को शांत करने और चिंता कम करने के लिए", "symptoms": ["चिंता", "बेचैनी", "अनिद्रा"]},
        "Kharaharapriya": {"benefit": "नर्वस चिड़चिड़ेपन में सहायता के लिए", "symptoms": ["चिड़चिड़ापन", "बेचैनी"]}
    },
    "kn": {
        "Kalyani": {"benefit": "ಶಕ್ತಿ ಹೆಚ್ಚಿಸಲು ಮತ್ತು ಭಯ ನಿವಾರಿಸಲು", "symptoms": ["ಆಯಾಸ", "ಆತಂಕ", "ಪ್ರೇರಣೆ ಕೊರತೆ"]},
        "Neelambari": {"benefit": "ನಿದ್ರಾಹೀನತೆ ಚಿಕಿತ್ಸೆ ಮತ್ತು ಶಾಂತ ನಿದ್ರೆಗಾಗಿ", "symptoms": ["ನಿದ್ರಾಹೀನತೆ", "ಚಡಪಡಿಕೆ"]},
        "Pantuvarali": {"benefit": "ಮಾನಸಿಕ ಗೊಂದಲ ಮತ್ತು ದುಃಖ ನಿವಾರಿಸಲು", "symptoms": ["ಖಿನ್ನತೆ", "ಪ್ರೇರಣೆ ಕೊರತೆ", "ಆತಂಕ"]},
        "Anandabhairavi": {"benefit": "ಹೊಟ್ಟೆ ನೋವು ಮತ್ತು ರಕ್ತದೊತ್ತಡ ನಿಯಂತ್ರಿಸಲು", "symptoms": ["ಆಮ್ಲೀಯತೆ", "ಉರಿಯೂತ"]},
        "Sahana": {"benefit": "ಅಧಿಕ ರಕ್ತದೊತ್ತಡ ನಿಯಂತ್ರಣ ಮತ್ತು ಮನಸ್ಸು ಶಾಂತ ಮಾಡಲು", "symptoms": ["ಕಿರಿಕಿರಿ", "ಆತಂಕ"]},
        "Hamsadhwani": {"benefit": "ಉದ್ವೇಗ ಕಡಿಮೆ ಮಾಡಲು ಮತ್ತು ಸಕಾರಾತ್ಮಕ ಚಿಂತನೆ ಬೆಳೆಸಲು", "symptoms": ["ಚಡಪಡಿಕೆ", "ಆತಂಕ", "ಕಿರಿಕಿರಿ"]},
        "Mohanam": {"benefit": "ಒತ್ತಡ, ತಲೆನೋವು ಮತ್ತು ಅಜೀರ್ಣ ನಿರ್ವಹಣೆಗೆ", "symptoms": ["ತಲೆನೋವು", "ಅನಿಯಮಿತ ಹಸಿವು", "ನಿಧಾನ ಚಯಾಪಚಯ"]},
        "Kapi": {"benefit": "ಖಿನ್ನತೆ ಮತ್ತು ನರ ಒತ್ತಡ ಕಡಿಮೆ ಮಾಡಲು", "symptoms": ["ಖಿನ್ನತೆ", "ಆತಂಕ", "ನಿದ್ರಾಹೀನತೆ"]},
        "Madhyamavati": {"benefit": "ನರ ಸಮಸ್ಯೆ ಮತ್ತು ದೇಹ ನೋವಿಗೆ", "symptoms": ["ಆಯಾಸ", "ದೇಹ ಭಾರವಾಗಿದೆ"]},
        "Yamuna Kalyani": {"benefit": "ಮಾನಸಿಕ ಚೈತನ್ಯಕ್ಕಾಗಿ", "symptoms": ["ಸೋಮಾರಿತನ", "ಆಯಾಸ", "ಪ್ರೇರಣೆ ಕೊರತೆ"]},
        "Bhairavi": {"benefit": "ದೀರ್ಘಕಾಲಿಕ ತಲೆನೋವು ಮತ್ತು ಉರಿಯೂತಕ್ಕೆ ಸಹಾಯ ಮಾಡಲು", "symptoms": ["ತಲೆನೋವು", "ಉರಿಯೂತ", "ಅತಿಯಾದ ಲೋಳೆ"]},
        "Revati": {"benefit": "ಇಂದ್ರಿಯಗಳನ್ನು ಶಾಂತಗೊಳಿಸಲು ಮತ್ತು ಆತಂಕ ಕಡಿಮೆ ಮಾಡಲು", "symptoms": ["ಆತಂಕ", "ಚಡಪಡಿಕೆ", "ನಿದ್ರಾಹೀನತೆ"]},
        "Kharaharapriya": {"benefit": "ನರ ಕಿರಿಕಿರಿಯಲ್ಲಿ ಸಹಾಯ ಮಾಡಲು", "symptoms": ["ಕಿರಿಕಿರಿ", "ಚಡಪಡಿಕೆ"]}
    }
}

vikriti_columns = [
    "Dry_skin", "Constipation", "Anxiety", "Insomnia", "Irregular_appetite", "Cold_intolerance", "Restlessness", "Fatigue",
    "Acidity", "Burning_sensation", "Irritability", "Sweating", "Headache", "Loose_stools", "Hunger", "Inflammation",
    "Lethargy", "Weight_gain", "Slow_metabolism", "Excess_sleep", "Depression", "Mucus", "Heavy_body", "Motivation_loss"
]

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

st.set_page_config(page_title="Ayurvedic Chatbot", layout="wide")

if 'p_idx' not in st.session_state: st.session_state.p_idx = 0
if 'p_answers' not in st.session_state: st.session_state.p_answers = {}
if 'lang' not in st.session_state: st.session_state.lang = "en"

# ── Language Selector ──
lang_display = st.selectbox(
    "🌐 Select Language / भाषा चुनें / ಭಾಷೆ ಆಯ್ಕೆ ಮಾಡಿ",
    list(LANGUAGES.keys()),
    index=list(LANGUAGES.values()).index(st.session_state.lang)
)
selected_lang = LANGUAGES[lang_display]

# Reset answers if language changed
if selected_lang != st.session_state.lang:
    st.session_state.lang = selected_lang
    st.session_state.p_idx = 0
    st.session_state.p_answers = {}
    st.rerun()

lang = st.session_state.lang
ui = UI_TEXT[lang]

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2 = st.tabs([ui["tab1"], ui["tab2"]])

# ── TAB 1: PRAKRITI ──
with tab1:
    st.title(ui["title1"])
    current_mappings = mappings[lang]
    p_feats = list(current_mappings.keys())

    if st.session_state.p_idx < len(p_feats):
        f = p_feats[st.session_state.p_idx]
        st.progress(st.session_state.p_idx / len(p_feats))

        if lang == "en":
            st.subheader(f"{ui['question_prefix']} {f}?")
        else:
            st.subheader(f"{ui['question_prefix']} {f} {ui.get('question_suffix', '?')}")

        current_options = list(current_mappings[f].values())
        saved_key = st.session_state.p_answers.get(f, None)
        default_idx = list(current_mappings[f].keys()).index(saved_key) if saved_key is not None else 0

        ans = st.radio(ui["choose_one"], current_options, index=default_idx, key=f"pr_{lang}_{st.session_state.p_idx}")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.session_state.p_idx > 0:
                if st.button(ui["prev"]):
                    st.session_state.p_answers[f] = [k for k, v in current_mappings[f].items() if v == ans][0]
                    st.session_state.p_idx -= 1
                    st.rerun()
        with col2:
            if st.button(ui["next"]):
                st.session_state.p_answers[f] = [k for k, v in current_mappings[f].items() if v == ans][0]
                st.session_state.p_idx += 1
                st.rerun()
    else:
        st.success(ui["complete"])
        if st.button(ui["view_result"]):
            try:
                with open('prakriti_model.pkl', 'rb') as f_model: m = pickle.load(f_model)
                with open('scaler.pkl', 'rb') as f_scaler: s = pickle.load(f_scaler)

                # Map localized answers back to numeric codes using English feature order
                en_feats = list(mappings["en"].keys())
                lang_feats = list(mappings[lang].keys())
                numeric_answers = {}
                for i, en_feat in enumerate(en_feats):
                    lang_feat = lang_feats[i]
                    numeric_answers[en_feat] = st.session_state.p_answers.get(lang_feat, 0)

                res = m.predict(s.transform(pd.DataFrame([numeric_answers])))[0]
                st.write(f"{ui['your_prakriti']} {p_lookup_lang[lang].get(res)}")
            except Exception as e:
                st.error(f"{ui['error_model']} {e}")

        if st.button(ui["reset"]):
            st.session_state.p_idx = 0
            st.session_state.p_answers = {}
            st.rerun()

# ── TAB 2: VIKRITI ──
with tab2:
    st.title(ui["title2"])

    phys_syms = physical_symptoms_lang[lang]
    ment_syms = mental_symptoms_lang[lang]
    rename_map = rename_map_lang[lang]

    st.subheader(ui["physical_symptoms"])
    sel_phys = st.multiselect(ui["select_physical"], phys_syms)
    st.subheader(ui["mental_symptoms"])
    sel_ment = st.multiselect(ui["select_mental"], ment_syms)

    if st.button(ui["analyze"]):
        try:
            with open('vikriti_model.pkl', 'rb') as f_vmodel: vm = pickle.load(f_vmodel)

            input_dict = {col: 0 for col in vikriti_columns}
            for sym in (sel_phys + sel_ment):
                if sym in rename_map:
                    input_dict[rename_map[sym]] = 1

            v_res = vm.predict(pd.DataFrame([input_dict])[vikriti_columns])[0]

            st.divider()
            st.error(f"{ui['aggravated_dosha']} {v_res}")

            plan = plan_templates_lang[lang][v_res]
            raga_data = raga_data_lang[lang]

            st.success(ui["recommendations"])

            c1, c2 = st.columns(2)
            with c1:
                st.subheader(ui["exercise_plan"])
                for ex in random.sample(plan["exercise"], min(len(plan["exercise"]), 3)):
                    st.write(f"- {ex}")
                st.subheader(ui["meditation"])
                st.write(plan["meditation"])
            with c2:
                st.subheader(ui["diet_plan"])
                for d in random.sample(plan["diet"], min(len(plan["diet"]), 3)):
                    st.write(f"- {d}")

                st.subheader(ui["music_therapy"])
                all_selected = sel_phys + sel_ment
                recommended_ragas = []
                for raga, info in raga_data.items():
                    if any(s in all_selected for s in info["symptoms"]):
                        if raga in plan["music"]:
                            recommended_ragas.append(f"**{raga}**: {ui['recommended']} {info['benefit']}.")

                if recommended_ragas:
                    for r in recommended_ragas[:2]:
                        st.write(f"- {r}")
                else:
                    random_fallback = random.sample(plan["music"], min(2, len(plan["music"])))
                    for r in random_fallback:
                        benefit = raga_data.get(r, {}).get("benefit", "")
                        st.write(f"- **{r}**: {ui['suggested']} {benefit}.")

        except Exception as e:
            st.error(f"{ui['prediction_error']} {e}")