import streamlit as st
import pandas as pd
from collections import defaultdict
import re
from datetime import date

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Decision Atlas", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.dim-card { background:#f0f4ff; border-left:4px solid #4472c4; padding:8px 12px;
            border-radius:4px; margin:4px 0; }
.step-card { background:#fafafa; border:1px solid #e0e0e0; padding:10px 14px;
             border-radius:6px; margin:6px 0; }
.step-label { font-weight:600; color:#333; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════════

# Helper: all fricative/affricate → stop pairs for Stopping detection
_FRICATIVES = ["f", "v", "s", "z", "th", "dh", "sh", "zh"]
_AFFRICATES = ["ch", "j"]
_STOPS = ["p", "b", "t", "d", "k", "g"]
_STOPPING_PAIRS = [(fric, stop) for fric in _FRICATIVES + _AFFRICATES for stop in _STOPS]

PROCESSES = {
    # ── SYLLABLE STRUCTURE ────────────────────────────────────────────────────
    "Weak Syllable Deletion": {
        "category": "Syllable structure", "dimension": "Syllable structure", "type": "word",
        "description": "Unstressed syllables are deleted from multisyllabic words, leaving only the stressed syllable(s).",
        "examples_word": [("banana","nana"), ("elephant","efant"), ("potato","tato"),
                          ("computer","puter"), ("tomato","mato"), ("umbrella","brella")],
        "development": "Typically resolves by ~3;0–3;5.",
        "resolution_age_months": 41,
        "clinical": "Affects intelligibility for multisyllabic words; can impact early word learning.",
        "therapy": ["Syllable segmentation and tapping activities",
                    "Auditory bombardment with multisyllabic targets",
                    "Multisyllabic minimal pairs"]
    },
    "Final Consonant Deletion": {
        "category": "Syllable structure", "dimension": "Coda stability", "type": "word",
        "description": "Consonants in word-final (coda) position are omitted, reducing CVC words to CV.",
        "examples_word": [("cat","ca"), ("cup","cu"), ("bed","be"), ("dog","do"),
                          ("duck","du"), ("frog","fro"), ("knife","nai"), ("chess","che")],
        "development": "Typically resolves by ~3;0.",
        "resolution_age_months": 36,
        "clinical": "Reduces intelligibility; collapses the CV/CVC contrast; impacts plural /z/, past tense /d/ morphology. Associated with hearing impairment (more errors on word-final consonants generally).",
        "therapy": ["Final consonant cueing (tactile, visual)",
                    "Auditory bombardment with CVC targets",
                    "Minimal pairs contrasting CV vs CVC (e.g. bee/bees)"]
    },
    "Cluster Reduction": {
        "category": "Syllable structure", "dimension": "Syllable structure", "type": "word",
        "description": "Consonant clusters are reduced to a single consonant, simplifying onset or coda complexity.",
        "examples_word": [("stop","top"), ("play","pay"), ("green","geen"), ("blue","boo"),
                          ("frog","fog"), ("sky","kai"), ("sport","pot"), ("spy","pai"),
                          ("snore","nor"), ("store","tor"), ("splash","plash"), ("string","ting")],
        "development": "2-element clusters resolve by ~3;6–3;11; 3-element clusters (e.g. /spl/, /str/) by ~4;0–4;5.",
        "resolution_age_months": 54,
        "clinical": "Affects onset complexity; reduces syllable shape contrasts. Distinguish carefully from final consonant deletion.",
        "therapy": ["Sequential blending of cluster elements",
                    "Minimal pairs contrasting cluster vs singleton (e.g. top/stop)",
                    "Hierarchy: /s/+stop before /s/+nasal before liquid clusters"]
    },
    "Epenthesis": {
        "category": "Syllable structure", "dimension": "Syllable structure", "type": "word",
        "description": "A vowel (often schwa) is inserted within a consonant cluster, adding an extra syllable.",
        "examples_word": [("blue","belue"), ("play","pelay"), ("green","gereen"), ("frog","ferog")],
        "development": "Typically resolves by ~3–4 years; often a transitional cluster acquisition strategy.",
        "resolution_age_months": 48,
        "clinical": "Transitional pattern during cluster acquisition; considered less marked than cluster reduction, as the cluster elements are still present.",
        "therapy": ["Auditory discrimination between cluster and epenthesis forms",
                    "Sequential blending activities"]
    },
    "Initial Consonant Deletion": {
        "category": "Other", "dimension": "Syllable structure", "type": "word",
        "description": "Consonants in word-initial (onset) position are omitted entirely.",
        "examples_word": [("cat","at"), ("sun","un"), ("dog","og"), ("fish","ish")],
        "development": "Atypical — not part of normal phonological development at any age.",
        "resolution_age_months": 0,
        "clinical": "Significant atypical pattern; markedly reduces intelligibility; warrants specialist assessment.",
        "therapy": ["Specialist phonological assessment",
                    "Initial consonant cueing (visual, tactile)",
                    "Auditory bombardment with initial-consonant targets"]
    },
    "Reduplication": {
        "category": "Syllable structure", "dimension": "Syllable structure", "type": "word",
        "description": "Complete or partial duplication of a stressed syllable replaces the full word form (e.g. a CVCV word becomes CVCV where both syllables are identical).",
        "examples_word": [("tiger","taitai"), ("bottle","bobo"), ("water","wawa"),
                          ("daddy","dada"), ("biscuit","bibi")],
        "development": "Typically resolves by ~3;0. One of the very earliest syllable structure processes.",
        "resolution_age_months": 36,
        "clinical": "One of the earliest simplification strategies; affects word shape and intelligibility in connected speech.",
        "therapy": ["Auditory bombardment with differentiated target word shapes",
                    "Syllable differentiation activities",
                    "Modelling varied CV/CVC structures"]
    },
    # ── SUBSTITUTION ──────────────────────────────────────────────────────────
    "Stopping": {
        "category": "Substitution", "dimension": "Manner contrasts", "type": "phoneme",
        "pairs": _STOPPING_PAIRS,
        "description": "Fricatives (/f, v, θ, ð, s, z, ʃ, ʒ/) and affricates (/tʃ, dʒ/) are replaced with plosives. The substituted stop may or may not match the target in voicing or place of articulation.",
        "examples_phoneme": [("f","p","fish → pish"), ("f","b","fish → bish"),
                              ("f","d","fish → dish"), ("s","t","sun → tun"),
                              ("s","d","sun → dun"), ("z","d","zoo → doo"),
                              ("sh","t","shoe → too"), ("sh","d","shoe → doo"),
                              ("th","t","thumb → tum"), ("ch","t","chair → tair"),
                              ("ch","d","chair → dair"), ("v","b","van → ban"),
                              ("v","d","van → dan")],
        "development": "Resolves for most fricatives by ~3;0. Dental fricatives /θ, ð/ may persist slightly longer.",
        "resolution_age_months": 36,
        "clinical": "Collapses the fricative/stop and affricate/stop manner contrasts; significantly reduces phoneme inventory. When stopping co-occurs with voicing or place errors, multiple processes are interacting simultaneously.",
        "therapy": ["Auditory discrimination of stops vs fricatives",
                    "Fricative placement cues (airflow awareness, frication feel on hand)",
                    "Minimal pairs (e.g. pie/fie, toe/so)",
                    "If combined with voicing errors, address stopping first before targeting voicing"]
    },
    "Velar Fronting": {
        "category": "Substitution", "dimension": "Place contrasts", "type": "phoneme",
        "pairs": [("k","t"),("g","d"),("ng","n"),("k","d"),("g","t")],
        "description": "Velar consonants /k, ɡ, ŋ/ are produced as alveolar consonants /t, d, n/ — the place of articulation moves forward.",
        "examples_phoneme": [("k","t","cat → tat"), ("k","d","cat → dat"),
                              ("g","d","go → do"), ("ng","n","ring → rin")],
        "development": "Typically resolves by ~3;6.",
        "resolution_age_months": 42,
        "clinical": "Loss of the velar/alveolar place contrast; one of the most common early phonological processes. Note: backing is atypical in English but velar fronting is typical.",
        "therapy": ["Minimal pairs targeting velar/alveolar contrast (e.g. tea/key, toe/go)",
                    "Velar placement cues (tongue-back elevation, 'back of tongue touching soft palate')",
                    "Phonological contrast therapy"]
    },
    "Palatal Fronting": {
        "category": "Substitution", "dimension": "Place contrasts", "type": "phoneme",
        "pairs": [("sh","s"),("zh","z"),("ch","ts"),("j","dz"),("sh","z"),("ch","s"),("j","z")],
        "description": "Postalveolar/palatal consonants /ʃ, ʒ, tʃ, dʒ/ are produced as alveolar consonants /s, z, ts, dz/ — place of articulation moves forward.",
        "examples_phoneme": [("sh","s","ship → sip"), ("zh","z","measure → meazure"),
                              ("ch","ts","chin → tsin"), ("j","dz","jump → dzump")],
        "development": "Typically resolves by ~4;0–4;5.",
        "resolution_age_months": 53,
        "clinical": "Loss of the palatal/alveolar place contrast.",
        "therapy": ["Auditory discrimination of palatal vs alveolar",
                    "Minimal pairs (e.g. sea/she, soup/shoe, chin/tin)",
                    "Articulatory cues for postalveolar position (tongue blade raised and retracted slightly)"]
    },
    "Gliding": {
        "category": "Substitution", "dimension": "Liquid contrasts", "type": "phoneme",
        "pairs": [("r","w"),("l","w"),("r","j"),("l","j"),("l","y"),("r","y")],
        "description": "Liquids /ɹ/ and /l/ are replaced with glides /w/ or /j/.",
        "examples_phoneme": [("r","w","red → wed"), ("r","w","rabbit → wabbit"),
                              ("l","w","leg → weg"), ("l","j","lamb → jamb"),
                              ("r","j","rabbit → jabbit")],
        "development": "/l/ typically resolves by ~5 years; /ɹ/ may persist until 6–7 years (within normal limits).",
        "resolution_age_months": 84,
        "clinical": "Liquid contrasts affected; /ɹ/ is one of the latest-developing English consonants. Gliding of approximants is a typical developmental process and is clinically distinct from gliding of fricatives, which is atypical.",
        "therapy": ["Motor learning approaches for /ɹ/ (if persistent after age 7)",
                    "Minimal pairs therapy (e.g. red/wed, lip/whip)",
                    "Phonological contrast therapy for /l/"]
    },
    "Gliding of Fricatives": {
        "category": "Other", "dimension": "Manner contrasts", "type": "phoneme",
        "pairs": [("f","w"),("v","w"),("th","w"),("dh","w"),
                  ("s","j"),("z","j"),("sh","j"),("zh","j"),
                  ("s","w"),("z","w"),("sh","w"),("zh","w"),
                  ("f","j"),("v","j"),("th","j"),("dh","j")],
        "description": "Fricatives (/f, v, θ, ð, s, z, ʃ, ʒ/) are replaced with glides /w/ or /j/. Clinically distinct from, and more significant than, gliding of approximants (/ɹ/ → /w/).",
        "examples_phoneme": [("f","w","fan → wan"), ("sh","j","should → jould"),
                              ("v","j","behave → behaje"), ("s","w","sun → wun"),
                              ("th","w","think → wink")],
        "development": "Atypical — not part of normal phonological development. Warrants specialist assessment.",
        "resolution_age_months": 0,
        "clinical": "Disordered pattern. Substantially reduces the fricative inventory. Much more clinically significant than gliding of approximants (/ɹ/ → /w/), which is a typical developmental process.",
        "therapy": ["Specialist phonological assessment",
                    "Detailed phonological process analysis (PPK, DEAP)",
                    "Consider multiple oppositions therapy",
                    "Fricative placement cues with strong auditory bombardment input"]
    },
    "Deaffrication": {
        "category": "Substitution", "dimension": "Manner contrasts", "type": "phoneme",
        "pairs": [("ch","sh"),("j","zh"),("ch","s"),("j","z"),("ch","f"),("j","v")],
        "description": "Affricates /tʃ, dʒ/ lose their stop (plosive) component and are produced as fricatives only.",
        "examples_phoneme": [("ch","sh","chair → share"), ("j","zh","jump → zhump"),
                              ("ch","s","cheese → seeze"), ("j","z","juice → zuice")],
        "development": "Typically resolves by ~4;0–4;5.",
        "resolution_age_months": 53,
        "clinical": "Collapses the affricate/fricative manner distinction; /tʃ/ and /ʃ/ become homophones.",
        "therapy": ["Auditory discrimination of affricate vs fricative",
                    "Blending stop + fricative approach (/tʃ/ = /t/+/ʃ/)",
                    "Minimal pairs (e.g. share/chair, zoo/juice)"]
    },
    "Affrication": {
        "category": "Substitution", "dimension": "Manner contrasts", "type": "phoneme",
        "pairs": [("s","ch"),("z","j"),("sh","ch"),("zh","j"),
                  ("t","ch"),("d","j"),("f","ch"),("v","j"),
                  ("th","ch"),("dh","j")],
        "description": "Fricatives or stops are replaced with affricates /tʃ/ or /dʒ/ — the reverse of deaffrication.",
        "examples_phoneme": [("s","ch","sun → chun"), ("z","j","zoo → joo"),
                              ("sh","ch","shoe → choo"), ("t","ch","tea → chea")],
        "development": "Less common than deaffrication; may resolve by ~4 years. Persistent affrication warrants assessment.",
        "resolution_age_months": 48,
        "clinical": "Reverse of deaffrication. Can co-occur with other manner changes. Reduces fricative/affricate and stop/affricate distinctions.",
        "therapy": ["Auditory discrimination of fricative/stop vs affricate",
                    "Minimal pairs contrasting affricate with target manner",
                    "Articulatory placement practice for sustained frication without stop onset"]
    },
    "Vocalisation": {
        "category": "Substitution", "dimension": "Liquid contrasts", "type": "phoneme",
        "pairs": [("l","u"),("l","o"),("l","uh"),("l","a"),("l","w"),("l","oo"),
                  ("r","uh"),("r","a"),("r","u"),("r","o"),("r","w"),("r","oo")],
        "description": "Syllabic or postvocalic liquids /l/ and /ɹ/ are replaced by vowels or vowel-like sounds, particularly in syllable-final or syllabic positions.",
        "examples_phoneme": [("l","u","bottle → bottou"), ("l","o","ball → baw"),
                              ("r","uh","butter → buttuh"), ("r","a","car → caa")],
        "development": "Typically resolves by ~5 years.",
        "resolution_age_months": 60,
        "clinical": "Primarily affects liquids in syllabic or word-final position; dark /l/ and syllabic /ɹ/ are particularly vulnerable.",
        "therapy": ["Syllabic liquid placement practice",
                    "Auditory discrimination of liquid vs vowel",
                    "Dark /l/ and syllabic /ɹ/ drill work"]
    },
    "Final Devoicing": {
        "category": "Voicing", "dimension": "Voicing contrasts", "type": "phoneme",
        "pairs": [("z","s"),("v","f"),("d","t"),("b","p"),("g","k"),("j","ch"),("zh","sh"),
                  ("dh","th")],
        "description": "Word-final (coda) voiced obstruents are produced as their voiceless counterparts.",
        "examples_phoneme": [("z","s","nose → noss"), ("d","t","bed → bet"),
                              ("b","p","cab → cap"), ("g","k","dog → dok"),
                              ("v","f","have → haf")],
        "development": "Typically resolves by ~3;0.",
        "resolution_age_months": 36,
        "clinical": "Collapses voicing contrasts in word-final position; may affect morphological markers (plural /-z/, past tense /-d/, third-person singular /-z/).",
        "therapy": ["Voiced/voiceless minimal pairs in final position (e.g. cap/cab, bat/bad)",
                    "Voicing awareness activities (feel laryngeal vibration)",
                    "Auditory bombardment with voiced-final targets"]
    },
    "Initial Voicing": {
        "category": "Voicing", "dimension": "Voicing contrasts", "type": "phoneme",
        "pairs": [("s","z"),("f","v"),("t","d"),("p","b"),("k","g"),("ch","j"),("sh","zh"),
                  ("th","dh")],
        "description": "Word-initial (onset) voiceless obstruents are produced as their voiced counterparts.",
        "examples_phoneme": [("s","z","sun → zun"), ("t","d","top → dop"),
                              ("p","b","pie → bie"), ("k","g","coat → goat"),
                              ("f","v","fan → van")],
        "development": "Typically resolves by ~3;0.",
        "resolution_age_months": 36,
        "clinical": "Collapses voicing contrasts in word-initial position. Related to prevocalic voicing — initial obstruents become voiced before a following vowel.",
        "therapy": ["Voiced/voiceless minimal pairs in initial position (e.g. pin/bin, tea/dee)",
                    "Aspiration and voice onset time (VOT) awareness activities"]
    },
    "Backing": {
        "category": "Other", "dimension": "Place contrasts", "type": "phoneme",
        "pairs": [("t","k"),("d","g"),("n","ng"),("s","k"),("s","g"),
                  ("t","g"),("d","k"),("s","h"),("z","h"),
                  ("t","q"),("d","q"),("s","q")],
        "description": "Alveolar or anterior consonants are produced at a more posterior place of articulation (velar, pharyngeal, or glottal) — the reverse of fronting.",
        "examples_phoneme": [("t","k","top → cop"), ("d","g","dog → gog"),
                              ("s","k","sun → kun"), ("s","h","sun → hun"),
                              ("n","ng","no → ŋo")],
        "development": "Atypical in English — not part of normal phonological development. Warrants assessment. (Note: typical in some languages, e.g. Cantonese.)",
        "resolution_age_months": 0,
        "clinical": "Atypical pattern; may indicate specific phonological disorder; associated with cleft palate/VPI (especially backing to velar, pharyngeal, or glottal places of articulation). Specialist referral recommended.",
        "therapy": ["Detailed phonological assessment",
                    "Specialist referral",
                    "Contrast therapy targeting anterior placement",
                    "If associated with cleft/VPI, coordinate with ENT/cleft palate team"]
    },
    "Denasalisation": {
        "category": "Other", "dimension": "Manner contrasts", "type": "phoneme",
        "pairs": [("m","b"),("n","d"),("ng","g"),("m","p"),("n","t"),("ng","k")],
        "description": "Nasal consonants /m, n, ŋ/ are replaced with their oral plosive counterparts — nasality is lost.",
        "examples_phoneme": [("m","b","man → ban"), ("n","d","no → do"),
                              ("ng","g","ring → rig"), ("m","p","man → pan"),
                              ("n","t","no → to")],
        "development": "Atypical — associated with cleft palate/VPI (causing hyponasality) or hearing impairment.",
        "resolution_age_months": 0,
        "clinical": "Disordered pattern; associated with structural anomalies (cleft, VPI) causing hyponasality, or with deafness (e.g. confusion of /b/ and /m/, or /n/ and /l/). Requires referral for velopharyngeal assessment.",
        "therapy": ["Velopharyngeal assessment referral",
                    "ENT/cleft palate team involvement",
                    "Nasal airflow awareness activities",
                    "Auditory discrimination of nasal /m, n, ŋ/ vs oral /b, d, ɡ/ consonants"]
    },
    "Nasalisation": {
        "category": "Other", "dimension": "Manner contrasts", "type": "phoneme",
        "pairs": [("b","m"),("d","n"),("g","ng"),("p","m"),("t","n"),("k","ng")],
        "description": "Oral plosives are replaced with nasal consonants — nasal airflow is present where it should not be (hypernasality / nasal escape).",
        "examples_phoneme": [("b","m","big → mig"), ("d","n","dog → nog"),
                              ("g","ng","go → ŋo"), ("p","m","pie → mie")],
        "development": "Atypical — associated with cleft palate/VPI (hypernasality/nasal escape on pressure consonants).",
        "resolution_age_months": 0,
        "clinical": "Disordered pattern; indicative of velopharyngeal insufficiency (VPI). Oral pressure consonants (/p, b, t, d, k, ɡ/, fricatives, affricates) require velopharyngeal closure; if the VP port is inadequate, airflow escapes nasally.",
        "therapy": ["Velopharyngeal assessment referral",
                    "ENT/cleft palate team involvement",
                    "Do NOT attempt oral motor exercises to correct VP closure — these are ineffective",
                    "Surgical or prosthetic VP management is typically required before speech therapy targets can be achieved"]
    },
    # ── ASSIMILATION ──────────────────────────────────────────────────────────
    "Progressive Assimilation": {
        "category": "Assimilation", "dimension": "Syllable structure", "type": "manual",
        "description": "A sound takes on features of a preceding sound in the same word — phonological harmony spreads left-to-right.",
        "examples_word": [("dog","dod"), ("top","pop"), ("cat","tat"), ("duck","dud")],
        "development": "Typically resolves by ~3;0.",
        "resolution_age_months": 36,
        "clinical": "Whole-word phonological harmony; can affect intelligibility across multisyllabic words.",
        "therapy": ["Auditory discrimination of target form vs assimilated form",
                    "Segmental approach to the word",
                    "Minimal pairs approach"]
    },
    "Regressive Assimilation": {
        "category": "Assimilation", "dimension": "Syllable structure", "type": "manual",
        "description": "A sound takes on features of a following sound in the same word — phonological harmony spreads right-to-left.",
        "examples_word": [("dog","gog"), ("top","kop"), ("cat","kak"), ("duck","guk")],
        "development": "Typically resolves by ~3;0.",
        "resolution_age_months": 36,
        "clinical": "Later segment influences earlier segment; a later consonant 'pulls' an earlier one towards its place or manner.",
        "therapy": ["Auditory discrimination training",
                    "Segmental approach to the word"]
    },
}

DIMENSIONS = {
    "Place contrasts": {
        "description": "Distinctions between consonants at different places of articulation (bilabial, labiodental, dental, alveolar, postalveolar, velar).",
        "processes": ["Velar Fronting", "Palatal Fronting", "Backing"],
        "significance": "Place contrasts form the basis of many consonant distinctions. Loss of place contrast reduces the phonemic inventory substantially."
    },
    "Manner contrasts": {
        "description": "Distinctions between consonants with different manners of production (plosives, fricatives, affricates, nasals, approximants, glides).",
        "processes": ["Stopping", "Deaffrication", "Affrication", "Gliding of Fricatives",
                      "Denasalisation", "Nasalisation"],
        "significance": "Manner contrasts distinguish plosives, fricatives, affricates, and nasals. Stopping is one of the most common early processes. Gliding of fricatives, denasalisation, and nasalisation are atypical patterns."
    },
    "Voicing contrasts": {
        "description": "Distinctions between voiced and voiceless consonants, maintained across the phoneme inventory in both onset and coda positions.",
        "processes": ["Final Devoicing", "Initial Voicing"],
        "significance": "Voicing contrasts affect the phoneme inventory and morphological markers (plural /-z/, past tense /-d/)."
    },
    "Liquid contrasts": {
        "description": "Production accuracy and contrast maintenance for the liquids /ɹ/ and /l/.",
        "processes": ["Gliding", "Vocalisation"],
        "significance": "Liquids are late-developing phonemes. /ɹ/ may not stabilise until age 6–7 in typically developing children, which is within normal limits."
    },
    "Syllable structure": {
        "description": "The integrity, complexity, and shape of syllables within words — including onset clusters, unstressed syllables, and word shape.",
        "processes": ["Cluster Reduction", "Epenthesis", "Weak Syllable Deletion",
                      "Initial Consonant Deletion", "Reduplication",
                      "Progressive Assimilation", "Regressive Assimilation"],
        "significance": "Syllable structure processes affect word shape, intelligibility, and the ability to produce complex onsets and multisyllabic words."
    },
    "Coda stability": {
        "description": "The presence and accuracy of consonants in word-final (coda) position.",
        "processes": ["Final Consonant Deletion"],
        "significance": "Coda stability affects intelligibility and morphological development (plural, past tense, possessive, third-person singular)."
    },
}

CLINICAL_ACTIONS = {
    "Further assessment": [
        "Administer DEAP for comprehensive phonological assessment.",
        "Complete inconsistency assessment (DEAP Section 5 or equivalent).",
        "Consider PPSA (Phonological Process Screening Assessment).",
        "Conduct connected speech sample analysis.",
        "If deafness-related patterns suspected, refer for audiological assessment.",
        "If cleft/VPI-related patterns suspected, refer for velopharyngeal assessment."
    ],
    "Consistency testing": [
        "Present same targets multiple times to check for consistent/inconsistent error patterns.",
        "Inconsistent disorder = variable productions of the same word across attempts.",
        "Use DEAP Inconsistency Assessment (25 words x 3 repetitions).",
        "Distinguish inconsistent phonological disorder from childhood apraxia of speech (CAS)."
    ],
    "Phonological contrast therapy": [
        "Minimal pairs — target error vs target sound in meaningful communicative context.",
        "Multiple oppositions — for children with multiple substitutions for one target sound.",
        "Maximal oppositions — contrast sounds differing on multiple phonological features.",
        "Empty set — contrast two sounds neither of which the child produces correctly."
    ],
    "Cycles approach": [
        "Suitable for highly unintelligible children with multiple phonological patterns.",
        "Target patterns in cycles; each pattern targeted for ~6 hours then cycled.",
        "Auditory bombardment is a key component of every session.",
        "Particularly effective for syllable structure and assimilation patterns."
    ],
    "Monitoring": [
        "If processes are age-appropriate, monitor over 3–6 months.",
        "Review at next scheduled contact.",
        "Provide parent guidance on facilitating phonological development.",
        "Consider re-assessment if concerns persist."
    ],
    "Specialist referral": [
        "Atypical patterns (e.g. Backing, Gliding of Fricatives, Initial Consonant Deletion) warrant specialist assessment.",
        "Denasalisation/Nasalisation patterns require ENT/cleft palate team referral.",
        "Consider referral to a phonological disorder specialist.",
        "Document and flag for MDT discussion if relevant.",
        "Discuss with supervising SLT."
    ],
}

# ═══════════════════════════════════════════════════════════════════════════════
# DETECTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def parse_age_months(age_str):
    s = str(age_str).strip()
    m = re.match(r'^(\d+)[;:.,](\d+)$', s)
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))
    m = re.match(r'^(\d+)$', s)
    if m:
        n = int(m.group(1))
        return n * 12 if n < 20 else n
    return None

def count_syllables(word):
    return max(1, len(re.findall(r'[aeiou]+', word.lower())))

def clean(word):
    return re.sub(r'[^a-zA-Z]', '', word).lower()

def ends_consonant(word):
    w = clean(word)
    return bool(w) and w[-1] not in 'aeiou'

def starts_consonant(word):
    w = clean(word)
    return bool(w) and w[0] not in 'aeiou'

def has_initial_cluster(word):
    w = clean(word)
    n = 0
    for ch in w:
        if ch not in 'aeiou':
            n += 1
        else:
            break
    return n >= 2

def is_reduplication(target, produced):
    tw, pw = clean(target), clean(produced)
    if len(pw) < 2 or len(tw) < 3:
        return False
    half = len(pw) // 2
    if len(pw) >= 4 and len(pw) % 2 == 0 and pw[:half] == pw[half:]:
        return True
    if len(pw) >= 4 and pw[0] == tw[0] and pw[:2] == pw[-(len(pw)//2):]:
        return True
    return False

def detect_phoneme(target, produced):
    t, p = target.lower().strip(), produced.lower().strip()
    if t == p:
        return []
    found = []
    for name, proc in PROCESSES.items():
        if proc.get("type") == "phoneme":
            for (pt, pp) in proc.get("pairs", []):
                if t == pt and p == pp:
                    found.append(name)
                    break
    return found

def detect_word(target, produced):
    tw, pw = clean(target), clean(produced)
    if tw == pw:
        return []
    found = []
    ts, ps = count_syllables(tw), count_syllables(pw)
    if is_reduplication(tw, pw):
        found.append("Reduplication")
    if ps < ts and ts >= 2:
        found.append("Weak Syllable Deletion")
    if ps > ts and has_initial_cluster(tw):
        found.append("Epenthesis")
    if ends_consonant(tw) and not ends_consonant(pw) and len(pw) < len(tw):
        found.append("Final Consonant Deletion")
    if starts_consonant(tw) and not starts_consonant(pw):
        found.append("Initial Consonant Deletion")
    if has_initial_cluster(tw) and not has_initial_cluster(pw) and ps <= ts:
        found.append("Cluster Reduction")
    return found

def run_analysis(observations):
    p_counts = defaultdict(int)
    p_examples = defaultdict(list)
    for obs in observations:
        otype = obs.get("type")
        if otype == "phoneme":
            procs = detect_phoneme(obs["target"], obs["produced"])
        elif otype in ("word", "stimulus"):
            procs = detect_word(obs["target"], obs["produced"])
        elif otype == "manual":
            procs = obs.get("manual_procs", [])
        else:
            procs = []
        obs["detected"] = procs
        for pr in procs:
            p_counts[pr] += 1
            p_examples[pr].append(obs)
    dims_hit = defaultdict(list)
    for pr in p_counts:
        dim = PROCESSES.get(pr, {}).get("dimension", "Other")
        if pr not in dims_hit[dim]:
            dims_hit[dim].append(pr)
    return dict(p_counts), dict(p_examples), dict(dims_hit)

def interpret(p_counts, age_months):
    if not p_counts:
        return None
    ok, concerns, atypical = [], [], []
    for pr in p_counts:
        ra = PROCESSES.get(pr, {}).get("resolution_age_months", 0)
        if ra == 0:
            atypical.append(pr)
        elif age_months and age_months <= ra:
            ok.append(pr)
        else:
            concerns.append(pr)
    n = len(p_counts)
    if atypical:
        label = "Atypical phonological pattern detected"
        colour = "error"
        msg = ("One or more atypical processes are present. These patterns are not expected "
               "at any developmental stage and warrant specialist phonological assessment.")
        actions = ["Specialist referral", "Further assessment"]
    elif concerns and n >= 3:
        label = "Multiple processes — possible phonological delay or disorder"
        colour = "warning"
        msg = (f"{n} processes detected, {len(concerns)} beyond the expected developmental range. "
               "A comprehensive phonological assessment is recommended.")
        actions = ["Further assessment", "Consistency testing", "Phonological contrast therapy"]
    elif concerns:
        label = f"{len(concerns)} process(es) beyond expected developmental range"
        colour = "warning"
        msg = "Some processes have persisted beyond their typical resolution age."
        actions = ["Further assessment", "Phonological contrast therapy", "Monitoring"]
    elif n >= 4:
        label = "Multiple age-appropriate processes"
        colour = "info"
        msg = "Multiple processes detected, all within developmental expectations. Monitor progress."
        actions = ["Monitoring", "Further assessment"]
    else:
        label = "Age-appropriate phonological processes"
        colour = "success"
        msg = "Detected processes are within the expected range for this age."
        actions = ["Monitoring"]
    return dict(label=label, colour=colour, msg=msg, ok=ok,
                concerns=concerns, atypical=atypical, actions=actions)

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM EFFECTS
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_EFFECTS = {
    "Weak Syllable Deletion":    "Reduces word length; collapses multisyllabic word shapes to stressed core.",
    "Final Consonant Deletion":  "Collapses CVC to CV; eliminates the coda contrast entirely.",
    "Cluster Reduction":         "Simplifies onset clusters; reduces syllable shape complexity.",
    "Epenthesis":                "Inserts schwa vowel into cluster; adds extra syllable.",
    "Initial Consonant Deletion":"Removes onset entirely; atypical collapse of word shape.",
    "Reduplication":             "Collapses polysyllabic words to a repeated syllable; very early developmental strategy.",
    "Stopping":                  "Fricatives and affricates collapse to plosives; collapses the manner contrast.",
    "Velar Fronting":            "Loss of velar /k, ɡ, ŋ/ vs alveolar /t, d, n/ place contrast.",
    "Palatal Fronting":          "Loss of postalveolar /ʃ, ʒ, tʃ, dʒ/ vs alveolar /s, z/ place contrast.",
    "Gliding":                   "Liquids /ɹ, l/ collapse into glides /w, j/; reduces liquid contrast.",
    "Gliding of Fricatives":     "Fricatives collapse into glides; atypical — much broader system impact than gliding of approximants.",
    "Deaffrication":             "Affricates /tʃ, dʒ/ lose stop element; collapses affricate/fricative distinction.",
    "Affrication":               "Fricatives/stops gain affricate quality; reduces manner distinctions.",
    "Vocalisation":              "Syllabic liquids /l, ɹ/ collapse to vowels; affects liquid inventory in final/syllabic position.",
    "Final Devoicing":           "Voiced codas devoiced; collapses the voicing contrast word-finally.",
    "Initial Voicing":           "Voiceless onsets voiced; collapses the voicing contrast word-initially.",
    "Backing":                   "Alveolars shift to velars/pharyngeals; atypical reversal of place contrast.",
    "Denasalisation":            "Nasals /m, n, ŋ/ collapse to oral plosives; atypical — associated with cleft/VPI/hearing loss.",
    "Nasalisation":              "Oral plosives collapse to nasals /m, n, ŋ/; atypical — associated with VP insufficiency.",
    "Progressive Assimilation":  "Earlier sound influences later sound; whole-word phonological harmony.",
    "Regressive Assimilation":   "Later sound influences earlier sound; whole-word phonological harmony.",
}

# ═══════════════════════════════════════════════════════════════════════════════
# CONSONANT CHART
# ═══════════════════════════════════════════════════════════════════════════════

CONSONANT_CHART = {
    "Stops":      {"bilabial": ["p","b"], "alveolar": ["t","d"], "velar": ["k","g"]},
    "Nasals":     {"bilabial": ["m"], "alveolar": ["n"], "velar": ["ng"]},
    "Fricatives": {"labiodental": ["f","v"], "dental": ["th","dh"],
                   "alveolar": ["s","z"], "postalveolar": ["sh","zh"]},
    "Affricates": {"postalveolar": ["ch","j"]},
    "Liquids":    {"alveolar": ["l","r"]},
    "Glides":     {"labial": ["w"], "palatal": ["y"]},
}

# ═══════════════════════════════════════════════════════════════════════════════
# PHONEME INVENTORY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def build_phoneme_inventory(observations):
    produced = set()
    targets  = set()
    for obs in observations:
        if obs.get("type") == "phoneme":
            t = obs["target"].strip().lower()
            p = obs["produced"].strip().lower()
            if t:
                targets.add(t)
            if p:
                produced.add(p)
    return {"produced": produced, "targets": targets, "missing": targets - produced}

def display_inventory_chart(inventory):
    produced = inventory["produced"]
    targeted = inventory["targets"]
    st.caption("🟢 Produced   🔴 Targeted but not produced   ⬜ Not in sample")
    for manner, places in CONSONANT_CHART.items():
        with st.expander(f"**{manner}**", expanded=True):
            cols = st.columns(len(places))
            for i, (place, phonemes) in enumerate(places.items()):
                with cols[i]:
                    st.caption(place)
                    for ph in phonemes:
                        if ph in produced:
                            st.markdown(
                                f"<span style='background:#d4edda;border-radius:4px;"
                                f"padding:2px 7px;font-weight:600'>🟢 {ph}</span>",
                                unsafe_allow_html=True)
                        elif ph in targeted:
                            st.markdown(
                                f"<span style='background:#f8d7da;border-radius:4px;"
                                f"padding:2px 7px'>🔴 {ph}</span>",
                                unsafe_allow_html=True)
                        else:
                            st.markdown(
                                f"<span style='color:#aaa;padding:2px 7px'>⬜ {ph}</span>",
                                unsafe_allow_html=True)

def inventory_reasoning(inventory):
    produced = inventory["produced"]
    targeted = inventory["targets"]
    insights = []
    velars = {"k", "g", "ng"}
    if velars & targeted and not velars & produced:
        insights.append(
            "**Velar inventory gap:** Velars /k, ɡ, ŋ/ were targeted but not produced. "
            "Consider whether this reflects Velar Fronting (a process) or true phoneme absence — "
            "the distinction affects therapy approach.")
    fricatives = {"f","v","s","z","sh","zh","th","dh"}
    if fricatives & targeted and not fricatives & produced:
        insights.append(
            "**Fricative inventory gap:** Fricatives /f, v, θ, ð, s, z, ʃ, ʒ/ targeted but absent from produced inventory. "
            "May reflect consistent Stopping or a broader inventory gap.")
    nasals = {"m", "n", "ng"}
    if nasals & targeted and not nasals & produced:
        insights.append(
            "**Nasal inventory gap:** Nasals /m, n, ŋ/ targeted but absent from produced inventory. "
            "May reflect Denasalisation — consider velopharyngeal or audiological assessment.")
    if produced and len(produced) < 6:
        insights.append(
            f"**Limited inventory:** Only {len(produced)} distinct consonant(s) produced in this sample "
            f"({', '.join(sorted(produced))}). Consider whether sample size is sufficient for inventory assessment.")
    return insights

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

for k, v in {
    "obs": [], "counter": 0,
    "child_age": "3;0", "test_used": "DEAP", "notes": "",
    "map_dim": None, "map_proc": None,
    "edu_proc": None, "edu_search_proc": None,
    "page": "input",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("Decision Atlas")
    st.caption("Phonological Analysis Tool")
    st.divider()
    st.subheader("Case Metadata")

    with st.form("meta"):
        age_in = st.text_input("Child Age", value=st.session_state.child_age,
                               placeholder="e.g. 3;6")
        tests = ["DEAP", "PPSA", "DLS", "DEAP-3", "EAT", "STAP", "CHIRPSA", "Other"]
        tidx = tests.index(st.session_state.test_used) if st.session_state.test_used in tests else 0
        test_in = st.selectbox("Test Used", tests, index=tidx)
        notes_in = st.text_area("Clinician Notes", value=st.session_state.notes,
                                height=80, placeholder="Optional...")
        if st.form_submit_button("Update", type="primary", use_container_width=True):
            st.session_state.child_age = age_in
            st.session_state.test_used = test_in
            st.session_state.notes = notes_in
            st.rerun()

    st.divider()
    if st.button("🏠  New Case", use_container_width=True, type="primary"):
        for _k in ["obs","counter","child_age","test_used","notes",
                   "map_dim","map_proc","edu_proc","edu_search_proc","page"]:
            st.session_state[_k] = {"obs":[],"counter":0,"child_age":"0;0",
                "test_used":"DEAP","notes":"","map_dim":None,"map_proc":None,
                "edu_proc":None,"edu_search_proc":None,"page":"input"}[_k]
        st.rerun()

    if st.button("Reset Session", use_container_width=True):
        st.session_state.obs = []
        st.session_state.counter = 0
        st.session_state.page = "input"
        st.rerun()

    age_m = parse_age_months(st.session_state.child_age)
    if age_m:
        st.caption(f"Age: {age_m} months ({age_m//12};{age_m%12:02d})")
    st.caption(f"Session: {date.today().strftime('%d %b %Y')}")

    st.divider()
    with st.expander("📖 Process Reference"):
        ref_proc = st.selectbox("Process", list(PROCESSES.keys()), key="sidebar_ref")
        rp = PROCESSES[ref_proc]
        st.markdown(f"**Description:** {rp['description']}")
        se = SYSTEM_EFFECTS.get(ref_proc, "")
        if se:
            st.markdown(f"**System effect:** {se}")
        st.markdown(f"**Development:** {rp['development']}")
        st.markdown(f"**Clinical note:** {rp['clinical']}")
        if rp.get("therapy"):
            st.markdown("**Therapy:**")
            for _t in rp["therapy"]:
                st.write(f"- {_t}")

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

age_months = parse_age_months(st.session_state.child_age)
if st.session_state.obs:
    p_counts, p_examples, dims_hit = run_analysis(st.session_state.obs)
    interp = interpret(p_counts, age_months)
    active_procs = set(p_counts.keys())
else:
    p_counts, p_examples, dims_hit, interp, active_procs = {}, {}, {}, None, set()

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.title("Decision Atlas")
c1, c2, c3 = st.columns(3)
c1.metric("Child Age", st.session_state.child_age)
c2.metric("Test", st.session_state.test_used)
c3.metric("Observations", len(st.session_state.obs))
if st.session_state.notes:
    st.info(f"**Notes:** {st.session_state.notes}")

# ═══════════════════════════════════════════════════════════════════════════════
# NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════

_PAGE_LABELS = ["Input & Observations", "Analysis & Reasoning",
                "System Map", "Educational Exploration", "Export"]
_PAGE_KEYS   = ["input", "analysis", "map", "edu", "export"]

_nav = st.radio("", _PAGE_LABELS,
                index=_PAGE_KEYS.index(st.session_state.page),
                horizontal=True, label_visibility="collapsed")
st.session_state.page = _PAGE_KEYS[_PAGE_LABELS.index(_nav)]
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — INPUT & OBSERVATIONS
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.page == "input":
    st.subheader("Add Observation")

    obs_type_choice = st.radio(
        "Observation type",
        ["Phoneme Substitution", "Word-Level Production", "Stimulus Item", "Manual Process Tag"],
        horizontal=True
    )

    with st.form("add_obs", clear_on_submit=True):
        if obs_type_choice == "Phoneme Substitution":
            ca, cb = st.columns(2)
            t_in = ca.text_input("Target phoneme", placeholder="k")
            p_in = cb.text_input("Produced phoneme", placeholder="t")
            s_in = ""
            otype = "phoneme"
            manual_sel = []

        elif obs_type_choice == "Word-Level Production":
            ca, cb = st.columns(2)
            t_in = ca.text_input("Target word", placeholder="banana")
            p_in = cb.text_input("Produced word", placeholder="nana")
            s_in = ""
            otype = "word"
            manual_sel = []

        elif obs_type_choice == "Stimulus Item":
            ca, cb, cc = st.columns(3)
            s_in = ca.text_input("Stimulus", placeholder="DEAP 17")
            t_in = cb.text_input("Target word", placeholder="banana")
            p_in = cc.text_input("Produced word", placeholder="nana")
            otype = "stimulus"
            manual_sel = []

        else:
            manual_sel = st.multiselect(
                "Select process(es) observed",
                [n for n, p in PROCESSES.items() if p.get("type") == "manual"]
            )
            t_in, p_in, s_in = "", "", ""
            otype = "manual"

        submitted = st.form_submit_button("Add Observation", type="primary")

    if submitted:
        if otype == "manual" and manual_sel:
            st.session_state.counter += 1
            st.session_state.obs.append({
                "id": st.session_state.counter, "type": "manual",
                "target": "—", "produced": "—", "stimulus": "",
                "manual_procs": manual_sel, "detected": manual_sel
            })
            st.rerun()
        elif t_in and p_in:
            st.session_state.counter += 1
            st.session_state.obs.append({
                "id": st.session_state.counter, "type": otype,
                "target": t_in.strip(), "produced": p_in.strip(),
                "stimulus": s_in.strip(), "detected": []
            })
            st.rerun()

    if st.session_state.obs:
        st.divider()
        st.subheader("Recorded Observations")

        hdr = st.columns([0.4, 1.8, 1.8, 1.8, 3.2, 0.5])
        for col, label in zip(hdr, ["#", "Target", "Produced", "Stimulus", "Detected Processes", ""]):
            col.markdown(f"**{label}**")
        st.markdown("---")

        to_del = None
        for obs in st.session_state.obs:
            row = st.columns([0.4, 1.8, 1.8, 1.8, 3.2, 0.5])
            row[0].write(obs["id"])
            row[1].write(f"**{obs['target']}**")
            row[2].write(obs["produced"])
            row[3].write(obs.get("stimulus") or "—")
            det = obs.get("detected", [])
            row[4].write(", ".join(det) if det else "—")
            if row[5].button("×", key=f"del_{obs['id']}"):
                to_del = obs["id"]

        if to_del is not None:
            st.session_state.obs = [o for o in st.session_state.obs if o["id"] != to_del]
            st.rerun()

        if p_counts:
            st.divider()
            st.subheader("Pattern Grouping")
            items = sorted(p_counts.items(), key=lambda x: -x[1])
            cols_pg = st.columns(min(len(items), 4))
            for i, (name, cnt) in enumerate(items):
                cols_pg[i % len(cols_pg)].metric(name, cnt)
    else:
        st.info("No observations yet. Use the form above to add observations.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ANALYSIS & REASONING
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == "analysis":
    if not st.session_state.obs:
        st.info("Add observations in **Input & Observations** to see analysis here.")
    else:
        st.subheader("Detected Phonological Processes")
        if not p_counts:
            st.warning("No processes automatically detected. Check your entries, or use Manual Process Tag for assimilation patterns.")
        else:
            for proc_name, cnt in sorted(p_counts.items(), key=lambda x: -x[1]):
                proc = PROCESSES.get(proc_name, {})
                ra = proc.get("resolution_age_months", 0)
                if ra == 0:
                    badge = "Atypical"
                elif age_months and age_months > ra:
                    badge = "Concern"
                else:
                    badge = "Age-appropriate"

                with st.expander(f"**{proc_name}** — {cnt} instance(s) — {badge}"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"**Description:** {proc.get('description','')}")
                        se = SYSTEM_EFFECTS.get(proc_name, "")
                        if se:
                            st.markdown(f"**System effect:** {se}")
                        st.markdown(f"**Development:** {proc.get('development','')}")
                        st.markdown(f"**Clinical note:** {proc.get('clinical','')}")
                        if proc.get("therapy"):
                            st.markdown("**Therapy approaches:**")
                            for t in proc["therapy"]:
                                st.write(f"- {t}")
                    with col_b:
                        st.markdown("**Your observations:**")
                        for ex_obs in p_examples.get(proc_name, []):
                            stim = f" ({ex_obs['stimulus']})" if ex_obs.get("stimulus") else ""
                            st.write(f"• {ex_obs['target']} → {ex_obs['produced']}{stim}")
                        if proc.get("examples_phoneme"):
                            st.markdown("**Typical examples:**")
                            for ep in proc["examples_phoneme"][:3]:
                                st.caption(ep[2])
                        elif proc.get("examples_word"):
                            st.markdown("**Typical examples:**")
                            for ew in proc["examples_word"][:3]:
                                st.caption(f"{ew[0]} → {ew[1]}")
                    _inv = build_phoneme_inventory(st.session_state.obs)
                    _prod = _inv["produced"]
                    if proc_name == "Velar Fronting" and not {"k","g"} & _prod:
                        st.warning("Velars absent from produced inventory. Consider phoneme absence rather than a process — this changes the therapy approach.")
                    elif proc_name == "Stopping" and not {"f","v","s","z","sh"} & _prod:
                        st.warning("Fricatives absent from produced inventory. May reflect an inventory gap rather than a process.")
                    elif proc_name == "Backing":
                        st.error("Backing is an atypical pattern. Flag for specialist review.")
                    elif proc_name == "Gliding of Fricatives":
                        st.error("Gliding of fricatives is an atypical/disordered pattern — distinct from gliding of approximants (/ɹ/ → /w/). Flag for specialist review.")
                    elif proc_name in ("Denasalisation", "Nasalisation"):
                        st.error(f"{proc_name} detected. Consider velopharyngeal assessment and ENT referral. Associated with cleft/VPI or hearing impairment.")

        st.divider()
        st.subheader("Phoneme Inventory")
        _inventory = build_phoneme_inventory(st.session_state.obs)
        if not _inventory["targets"]:
            st.caption("Add Phoneme Substitution observations to populate the inventory chart.")
        else:
            display_inventory_chart(_inventory)
            _insights = inventory_reasoning(_inventory)
            if _insights:
                st.markdown("**Inventory interpretation:**")
                for _ins in _insights:
                    st.info(_ins)

        st.divider()
        st.subheader("System Dimensions Affected")
        if dims_hit:
            dcols = st.columns(min(len(dims_hit), 3))
            for i, (dim, procs) in enumerate(dims_hit.items()):
                with dcols[i % 3]:
                    st.markdown(f"""
                    <div class="dim-card">
                        <div class="step-label">{dim}</div>
                        <div style="font-size:0.9em">{', '.join(procs)}</div>
                    </div>""", unsafe_allow_html=True)

        st.divider()
        st.subheader("Reasoning Pathway")

        obs_preview = ", ".join(
            f"{o['target']} → {o['produced']}" for o in st.session_state.obs[:6]
        ) + ("..." if len(st.session_state.obs) > 6 else "")
        st.markdown(f"""<div class="step-card">
            <div class="step-label">Step 1 — Observation</div>
            {len(st.session_state.obs)} observation(s) recorded: {obs_preview}
        </div>""", unsafe_allow_html=True)

        if p_counts:
            proc_summary = ", ".join(f"{n} ({c})" for n, c in
                                     sorted(p_counts.items(), key=lambda x: -x[1]))
            st.markdown(f"""<div class="step-card">
                <div class="step-label">Step 2 — Detected Processes</div>
                {proc_summary}
            </div>""", unsafe_allow_html=True)

            dim_summary = ", ".join(dims_hit.keys()) if dims_hit else "None identified"
            st.markdown(f"""<div class="step-card">
                <div class="step-label">Step 3 — System Dimensions Affected</div>
                {dim_summary}
            </div>""", unsafe_allow_html=True)

            if age_months:
                late = [n for n in p_counts if
                        PROCESSES.get(n, {}).get("resolution_age_months", 0) not in (0,) and
                        age_months > PROCESSES[n]["resolution_age_months"]]
                atyp = [n for n in p_counts if PROCESSES.get(n, {}).get("resolution_age_months", 0) == 0]
                if atyp:
                    dev_txt = f"Atypical processes detected: {', '.join(atyp)}."
                elif late:
                    dev_txt = (f"At age {st.session_state.child_age} ({age_months} months): "
                               f"{', '.join(late)} are beyond typical resolution age.")
                else:
                    dev_txt = (f"At age {st.session_state.child_age} ({age_months} months): "
                               "all detected processes are within developmental expectations.")
                st.markdown(f"""<div class="step-card">
                    <div class="step-label">Step 4 — Developmental Interpretation</div>
                    {dev_txt}
                </div>""", unsafe_allow_html=True)

        st.divider()
        st.subheader("System-Level Interpretation")
        if interp:
            colour = interp["colour"]
            body = f"**{interp['label']}**\n\n{interp['msg']}"
            if colour == "error":
                st.error(body)
            elif colour == "warning":
                st.warning(body)
            elif colour == "success":
                st.success(body)
            else:
                st.info(body)

            ca, cb, cc = st.columns(3)
            with ca:
                if interp["ok"]:
                    st.markdown("**Age-appropriate**")
                    for p in interp["ok"]:
                        st.success(f"✓ {p}")
            with cb:
                if interp["concerns"]:
                    st.markdown("**Concerns**")
                    for p in interp["concerns"]:
                        st.warning(f"⚠ {p}")
            with cc:
                if interp["atypical"]:
                    st.markdown("**Atypical**")
                    for p in interp["atypical"]:
                        st.error(f"✗ {p}")

            st.caption("These are clinical suggestions, not diagnoses. Professional judgement is required.")

        st.divider()
        st.subheader("Possible Next Steps")
        if interp:
            for action in interp["actions"]:
                with st.expander(f"**{action}**"):
                    for step in CLINICAL_ACTIONS.get(action, []):
                        st.write(f"- {step}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — SYSTEM MAP  (button highlighting fixed)
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == "map":
    st.subheader("Phonological System Map")
    st.caption("Select a dimension to explore. The selected dimension turns blue; detected processes are marked 🔴.")

    if active_procs:
        st.markdown(f"**Active in session:** {', '.join(sorted(active_procs))}")

    dim_names = list(DIMENSIONS.keys())
    dim_cols = st.columns(len(dim_names))
    for i, dim in enumerate(dim_names):
        dim_active = any(p in active_procs for p in DIMENSIONS[dim]["processes"])
        is_selected_dim = st.session_state.map_dim == dim
        label = f"{'🔴 ' if dim_active else ''}{dim}"
        with dim_cols[i]:
            # primary = selected (blue), secondary = not selected (grey)
            if st.button(label, key=f"map_{dim}", use_container_width=True,
                         type="primary" if is_selected_dim else "secondary"):
                if is_selected_dim:
                    st.session_state.map_dim = None
                    st.session_state.map_proc = None
                else:
                    st.session_state.map_dim = dim
                    st.session_state.map_proc = None
                st.rerun()

    if st.session_state.map_dim:
        dim = st.session_state.map_dim
        dinfo = DIMENSIONS[dim]
        st.divider()
        st.markdown(f"### {dim}")
        st.write(dinfo["description"])
        st.info(f"**Clinical significance:** {dinfo['significance']}")
        st.markdown("**Processes:**")

        proc_cols = st.columns(min(len(dinfo["processes"]), 3))
        for j, pname in enumerate(dinfo["processes"]):
            is_active = pname in active_procs
            is_selected_proc = st.session_state.map_proc == pname
            cnt = p_counts.get(pname, 0)
            # Show count if detected; mark selected with primary colour
            btn_label = f"{'🔴 ' if is_active else ''}{pname}{f' ({cnt})' if is_active else ''}"
            with proc_cols[j % 3]:
                if st.button(btn_label, key=f"mproc_{pname}",
                             type="primary" if is_selected_proc else "secondary",
                             use_container_width=True):
                    st.session_state.map_proc = None if is_selected_proc else pname
                    st.rerun()

        if st.session_state.map_proc and st.session_state.map_proc in dinfo["processes"]:
            pname = st.session_state.map_proc
            proc = PROCESSES.get(pname, {})
            st.divider()
            st.markdown(f"#### {pname}")
            pa, pb = st.columns(2)
            with pa:
                st.markdown(f"**Explanation:** {proc.get('description','')}")
                st.markdown(f"**Development:** {proc.get('development','')}")
                se = SYSTEM_EFFECTS.get(pname, "")
                if se:
                    st.markdown(f"**System effect:** {se}")
                st.markdown(f"**Clinical note:** {proc.get('clinical','')}")
                if proc.get("therapy"):
                    st.markdown("**Therapy approaches:**")
                    for t in proc["therapy"]:
                        st.write(f"- {t}")
            with pb:
                if proc.get("examples_phoneme"):
                    st.markdown("**Example errors:**")
                    for ep in proc["examples_phoneme"]:
                        st.write(f"- {ep[0]} → {ep[1]}: *{ep[2]}*")
                elif proc.get("examples_word"):
                    st.markdown("**Example productions:**")
                    for ew in proc["examples_word"]:
                        st.write(f"- {ew[0]} → {ew[1]}")
                if pname in active_procs:
                    st.success(f"**Detected in current session** ({p_counts[pname]} instances)")
                    for obs_ex in p_examples.get(pname, []):
                        st.write(f"- {obs_ex['target']} → {obs_ex['produced']}"
                                 + (f" ({obs_ex['stimulus']})" if obs_ex.get("stimulus") else ""))

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — EDUCATIONAL EXPLORATION
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == "edu":
    st.subheader("Educational Exploration Mode")
    st.markdown("Start from a **system dimension**, or search across all processes by name, symptom, or therapy approach.")

    search_q = st.text_input(
        "Search",
        placeholder="e.g. velar, ŋ, ʃ, minimal pairs, cleft, atypical, 3 years, stopping...",
        label_visibility="collapsed"
    )

    def build_search_corpus(pname, proc):
        parts = [
            pname,
            proc.get("description", ""),
            proc.get("clinical", ""),
            proc.get("development", ""),
            proc.get("category", ""),
            proc.get("dimension", ""),
            " ".join(proc.get("therapy", [])),
            " ".join(ep[2] for ep in proc.get("examples_phoneme", [])),
            " ".join(f"{ew[0]} {ew[1]}" for ew in proc.get("examples_word", [])),
        ]
        return " ".join(parts).lower()

    def render_process_panel(pname, source_key):
        proc = PROCESSES.get(pname, {})
        dim = proc.get("dimension", "")
        st.markdown(f"## {pname}")
        st.markdown(f"*{proc.get('category', '')} — {dim}*")
        st.divider()
        st.markdown(f"**Explanation:** {proc.get('description', '')}")
        st.markdown(f"**Developmental timeline:** {proc.get('development', '')}")
        se = SYSTEM_EFFECTS.get(pname, "")
        if se:
            st.markdown(f"**System effect:** {se}")
        st.markdown(f"**Clinical note:** {proc.get('clinical', '')}")
        if proc.get("examples_phoneme"):
            st.markdown("**Example errors:**")
            df_ex = pd.DataFrame(proc["examples_phoneme"], columns=["Target", "Produced", "Example"])
            st.table(df_ex)
        elif proc.get("examples_word"):
            st.markdown("**Example productions:**")
            df_ex = pd.DataFrame(proc["examples_word"], columns=["Target", "Produced"])
            st.table(df_ex)
        if proc.get("therapy"):
            st.markdown("**Therapy approaches:**")
            for t in proc["therapy"]:
                st.write(f"- {t}")
        if pname in active_procs:
            st.success(f"**Active in current session** — {p_counts[pname]} instance(s)")

    if search_q.strip():
        q = search_q.strip().lower()
        matches = [
            pname for pname, proc in PROCESSES.items()
            if q in build_search_corpus(pname, proc)
        ]
        col_l, col_r = st.columns([1, 2])
        with col_l:
            if matches:
                st.markdown(f"**{len(matches)} result(s) for** *\"{search_q}\"*")
                for pname in matches:
                    is_active = pname in active_procs
                    dim_tag = PROCESSES[pname].get("dimension", "")
                    label = f"{'🔴 ' if is_active else ''}→ {pname}"
                    st.caption(dim_tag)
                    if st.button(label, key=f"srch_{pname}", use_container_width=True):
                        st.session_state.edu_search_proc = pname
                        st.rerun()
            else:
                st.warning(f"No processes matched *\"{search_q}\"*.")
                st.caption("Try: process name, IPA symbol, symptom, therapy approach, or developmental age.")
        with col_r:
            selected = st.session_state.edu_search_proc
            if selected and selected in matches:
                render_process_panel(selected, "search")
            elif matches:
                st.info("Select a result on the left to view its full knowledge panel.")
    else:
        col_l, col_r = st.columns([1, 2])
        with col_l:
            edu_dim = st.selectbox(
                "Select a system dimension",
                ["— Select —"] + list(DIMENSIONS.keys())
            )
            if edu_dim != "— Select —":
                dinfo = DIMENSIONS[edu_dim]
                st.info(dinfo["description"])
                st.markdown("**Processes:**")
                for pname in dinfo["processes"]:
                    is_active = pname in active_procs
                    label = f"{'🔴 ' if is_active else ''}→ {pname}"
                    if st.button(label, key=f"edu_{pname}", use_container_width=True):
                        st.session_state.edu_proc = pname
                        st.rerun()
        with col_r:
            if edu_dim != "— Select —" and st.session_state.edu_proc:
                pname = st.session_state.edu_proc
                if pname in DIMENSIONS[edu_dim]["processes"]:
                    render_process_panel(pname, "browse")
                else:
                    st.info("Select a process from the left.")
            elif edu_dim != "— Select —":
                st.info("Select a process from the left panel.")
            else:
                st.info("Select a system dimension from the dropdown to begin.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — EXPORT
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == "export":
    st.subheader("Case Summary Export")
    if not st.session_state.obs:
        st.info("Add observations to generate a case summary.")
    else:
        lines = [
            "DECISION ATLAS — CASE SUMMARY",
            "=" * 44,
            f"Date:       {date.today().strftime('%d %b %Y')}",
            f"Child Age:  {st.session_state.child_age}",
            f"Test Used:  {st.session_state.test_used}",
        ]
        if st.session_state.notes:
            lines.append(f"Notes:      {st.session_state.notes}")
        lines += ["", f"OBSERVATIONS ({len(st.session_state.obs)})", "-" * 44]
        for obs in st.session_state.obs:
            stim = f" [{obs['stimulus']}]" if obs.get("stimulus") else ""
            det = f" → {', '.join(obs['detected'])}" if obs.get("detected") else ""
            lines.append(f"  {obs['target']} → {obs['produced']}{stim}{det}")
        if p_counts:
            lines += ["", "DETECTED PROCESSES", "-" * 44]
            for pname, cnt in sorted(p_counts.items(), key=lambda x: -x[1]):
                ra = PROCESSES.get(pname, {}).get("resolution_age_months", 0)
                if ra == 0:
                    status = "(Atypical)"
                elif age_months and age_months > ra:
                    status = f"(Concern — typical resolution: {ra//12};{ra%12:02d})"
                else:
                    status = "(Age-appropriate)"
                lines.append(f"  {pname}: {cnt} instance(s) {status}")
        if dims_hit:
            lines += ["", "SYSTEM DIMENSIONS AFFECTED", "-" * 44]
            for dim, procs in dims_hit.items():
                lines.append(f"  {dim}: {', '.join(procs)}")
        if interp:
            lines += ["", "INTERPRETATION", "-" * 44,
                      f"  {interp['label']}", f"  {interp['msg']}",
                      "", "SUGGESTED NEXT STEPS", "-" * 44]
            for a in interp["actions"]:
                lines.append(f"  • {a}")
        lines += ["", "=" * 44,
                  "Note: For clinical support only. Professional judgement required for all decisions."]

        summary = "\n".join(lines)
        st.text_area("Summary", summary, height=380)
        st.download_button("Download (.txt)", data=summary,
                           file_name=f"decision_atlas_{date.today().strftime('%Y%m%d')}.txt",
                           mime="text/plain", type="primary")

        if p_counts:
            st.divider()
            st.markdown("**Process Table**")
            rows = []
            for pname, cnt in sorted(p_counts.items(), key=lambda x: -x[1]):
                ra = PROCESSES.get(pname, {}).get("resolution_age_months", 0)
                if ra == 0:
                    status = "Atypical"
                elif age_months and age_months > ra:
                    status = "Concern"
                else:
                    status = "Age-appropriate"
                rows.append({
                    "Process": pname, "Count": cnt,
                    "Dimension": PROCESSES.get(pname, {}).get("dimension", ""),
                    "Status": status,
                    "Typical resolution": PROCESSES.get(pname, {}).get("development", "")
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
