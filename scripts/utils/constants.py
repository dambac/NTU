"""
This is a file for most of paths and other constants used throughout a project.
"""

"""
NOTE: Fill those values yourself after acquiring NTU dataset
"""
ORIGINAL_VIDEOS_DIR = "/home/dbaciur/NTU/NTU/data/Processed"
ORIGINAL_ANNOTATIONS_FILE = "home/dbaciur/NTU/NTU/data/all_observations.tsv"
ERRATA_ANNOTATIONS_FILE = "home/dbaciur/NTU/NTU/coders_annotations_v_0.csv"
# manual annotations for video layout types prepared in previous works
LAYOUT_ANNOTATIONS_DIR = f"home/dbaciur/NTU/NTU/data//TheLastLabels"

# Directory where results of calculations should be stored
OUTPUT_DIR = "./output"


class C:
    OUTPUT_DIR = OUTPUT_DIR

    """
    ==========================================
    Output for annotations
    ==========================================
    """
    # annotations after applying errata and some pre processing
    PROCESSED_ANNOTATIONS_PATH = f"{OUTPUT_DIR}/annotations/processed_observations.csv"
    # annotations for only visual events
    VISUAL_EVENTS_PATH = f"{OUTPUT_DIR}/annotations/visual_events.csv"

    """
    ==========================================
    Output for frames
    ==========================================
    """

    class Frames:
        # root directory for screenshots of frames
        FRAMES_SCREENSHOTS_PATH = f"{OUTPUT_DIR}/frames/screenshots"
        # initial list of frames
        FRAMES_DF_DRAFT_PATH = f"{OUTPUT_DIR}/frames/frames_first_version.csv"
        # list of cleaned frames + layouts
        FRAMES_DF_PATH = f"{OUTPUT_DIR}/frames/frames_valid.csv"

        FRAMES_SPLIT_SCREENSHOTS_PATH = f"{OUTPUT_DIR}/frames/split_screenshots"

        POINT_DUPLICATES = f"{OUTPUT_DIR}/frames/point_frames_duplicated.csv"
        POINT = f"{OUTPUT_DIR}/frames/point_frames.csv"

        WEBSITE_S10_DUPLICATES = f"{OUTPUT_DIR}/frames/websites_10s_duplicated.csv"
        WEBSITE_S10 = f"{OUTPUT_DIR}/frames/websites_10s.csv"

        CHART_IMAGE_S10_LABEL_0 = f"{OUTPUT_DIR}/frames/chart_image_10s_label_0.csv"

    SAMPLES_DATAFRAMES_DIR = f"{OUTPUT_DIR}/samples"

    class MiscPaths:
        STATE_POINT_CROSSINGS_DF_PATH = f"{OUTPUT_DIR}/data/state_point_df.csv"

    class StatePoint:
        RANDOM_FRAMES_DIR = f"{OUTPUT_DIR}/analysis/random_frames"
        CHARTS_OUTPUT_DIR = "f{OUTPUT_DIR}/statepoint"
        GET_EVENT_FRAME_NAME = lambda event: f"{event.video}_{event.event_id}_{event.start}_{event.stop}.jpg"
        GET_VID_EVENT_FRAME_NAME = lambda event: f"{event.event_id}_{event.beh_id}_{event.start}-{event.stop}.jpg"

    class EventCrossings:
        DF_OUTPUT_DIR = "outputs/event_crossings"
        EVENT_LAYOUT_DISTRIBUTION_DIR = "outputs/layout_distributions"
        SCREEN_TIME_IMAGES_DIR = "outputs/screen_time"

    class DistributionsAndSets:
        DISTRIBUTIONS_RESULTS = f"{OUTPUT_DIR}/analysis/distributions"
        SETS_SPLITS = f"{OUTPUT_DIR}/sets_split"
        SETS_SPLITS_SAMPLES = f"{OUTPUT_DIR}/sets_split/samples"

    ALEX_NET_M2_PATH = f"{OUTPUT_DIR}/transfer_views/AlexNetViews"
    ALEX_NET_PATH = f"{OUTPUT_DIR}/transfer_views/DeepAlexNetViews"
    RES_NET_PATH = f"{OUTPUT_DIR}/transfer_views/ResNetViews"
    VGG_M2_PATH = f"{OUTPUT_DIR}/transfer_views/VggViews"
    VGG_PATH = f"{OUTPUT_DIR}/transfer_views/DeepVggViews"

    RESULTS_DIR = f"{OUTPUT_DIR}/results"
    RESULTS_ANALYSIS_DIR = f"{OUTPUT_DIR}/results_analysis"

    # Annotations columns
    A_OBSERVATION_ID = 'Observation id'
    A_OBSERVATION_DATE = 'Observation date'
    A_MEDIA_FILE = 'Media file'
    A_TOTAL_LENGTH = 'Total length'
    A_FPS = 'FPS'
    A_SUBJECT = 'Subject'
    A_BEHAVIOR = 'Behavior'
    A_BEHAVIORAL_CATEGORY = 'Behavioral category'
    A_MODIFIERS = 'Modifiers'
    A_BEHAVIOR_TYPE = "Behavior type"
    A_START = 'Start (s)'
    A_STOP = 'Stop (s)'
    A_DURATION = 'Duration (s)'
    A_COMMENT_START = 'Comment start'
    A_COMMENT_STOP = 'Comment stop'
    A_LECTURE_ID = "lecture_id"
    A_RA_ID = "research_assistant_id"
    A_SUBJECT_ID = "subject_id"
    # new columns
    A_EVENT_ID = "Id"
    A_BEH_ID = "BehaviorId"
    A_VIDEO = "Video"
    A_TIME = "Time"

    # to avoid weird precission changes
    A_CONVERTERS = {
        A_START: lambda value: float(str(value)),
        A_STOP: lambda value: float(str(value)),
        A_TIME: lambda value: float(str(value)),
        A_DURATION: lambda value: float(str(value)),
    }

    B_EYE = "B_EYE"
    B_WRITING_SLIDES = "B_WRITING_SLIDES"
    B_VOICE = "B_VOICE"
    B_ASK_QUESTIONS_S = "B_ASK_QUESTIONS_S"
    B_ASK_QUESTIONS_P = "B_ASK_QUESTIONS_P"
    B_GIVE_QUESTIONS_S = "B_GIVE_QUESTIONS_S"
    B_GIVE_QUESTIONS_P = "B_GIVE_QUESTIONS_P"
    B_CHARTS_S = "B_CHARTS_S"
    B_CHARTS_P = "B_CHARTS_P"
    B_ORGANIZATION_S = "B_ORGANIZATION_S"
    B_ORGANIZATION_P = "B_ORGANIZATION_P"
    B_LAUGHTER = "B_LAUGHTER"
    B_DEMONSTRATION = "B_DEMONSTRATION"
    B_IMAGES_S = "B_IMAGES_S"
    B_IMAGES_P = "B_IMAGES_P"
    B_WEBSITE_S = "B_WEBSITE_S"
    B_WEBSITE_P = "B_WEBSITE_P"
    B_STANDS = "B_STANDS"
    B_MOVEMENT = "B_MOVEMENT"
    B_WRITING_BOARD = "B_WRITING_BOARD"
    B_SUMMING_UP = "B_SUMMING_UP"
    B_DISCIPLINE = "B_DISCIPLINE"
    B_FILMS_S = "B_FILMS_S"
    B_FILMS_P = "B_FILMS_P"
    B_ASSIGN = "B_ASSIGN"
    B_STUDENT_QUESTION_S = "B_STUDENT_QUESTION_S"
    B_STUDENT_QUESTION_P = "B_STUDENT_QUESTION_P"
    B_HINTS_S = "B_HINTS_S"
    B_HINTS_P = "B_HINTS_P"
    B_BIBLIO_S = "B_BIBLIO_S"
    B_BIBLIO_P = "B_BIBLIO_P"
    B_SESSIONS_S = "B_SESSIONS_S"
    B_SESSIONS_P = "B_SESSIONS_P"
    B_DISCUSSION = "B_DISCUSSION"
    B_EMPTY = "B_EMPTY"

    # Behaviors
    BEH_IDS = {
        'eye contact': B_EYE,
        'Writing on slides': B_WRITING_SLIDES,
        'use of voice intonation to Emphasise more important issues/topics': B_VOICE,
        'S, Asking questions': B_ASK_QUESTIONS_S,
        'P, Asking questions': B_ASK_QUESTIONS_P,
        'S, giving questions to students: rhetorical, comprehension questions': B_GIVE_QUESTIONS_S,
        'P, giving questions to students: rhetorical, comprehension questions", point each question': B_GIVE_QUESTIONS_P,
        'S, cHarts in slides': B_CHARTS_S,
        'P, cHarts in slides': B_CHARTS_P,
        'S, organization: giving class outline, clearly indicating Transition from one topic to another': B_ORGANIZATION_S,
        'P, organization: giving class outline, clearly indicating Transition from one topic to another"': B_ORGANIZATION_P,
        'Laughter': B_LAUGHTER,
        'demonstration': B_DEMONSTRATION,
        'S, Images in slides': B_IMAGES_S,
        'P, Images in slides': B_IMAGES_P,
        'S, website': B_WEBSITE_S,
        'P, website': B_WEBSITE_P,
        'Active teacher stands by slides and explains them': B_STANDS,
        'Movement across podium': B_MOVEMENT,
        'Writing on a whiteboard': B_WRITING_BOARD,
        'summing Up': B_SUMMING_UP,
        'discipline': B_DISCIPLINE,
        'S, Films or animations in slides': B_FILMS_S,
        'P, Films or animations in slides': B_FILMS_P,
        'assignments': B_ASSIGN,
        'S, students are asking questions, generating their own ideas': B_STUDENT_QUESTION_S,
        'P, students are asking questions, generating their own ideas': B_STUDENT_QUESTION_P,
        'S, giving hints how to do something': B_HINTS_S,
        'P, giving hints how to do something': B_HINTS_P,
        'S, Referring to bibliography, other researchers': B_BIBLIO_S,
        'P, Referring to bibliography, other researchers': B_HINTS_P,
        'S, Session on tests': B_SESSIONS_S,
        'P, Session on tests': B_SESSIONS_P,
        "students' Disscusion": B_DISCUSSION,
        'empty': B_EMPTY
    }
    BEH_BY_ID = {v: k for k, v in BEH_IDS.items()}
    ALL_BEHS = list(BEH_IDS.values())

    POINT_BEHS = [
        B_ASK_QUESTIONS_P,
        B_GIVE_QUESTIONS_P,
        B_CHARTS_P,
        B_ORGANIZATION_P,
        B_IMAGES_P,
        B_WEBSITE_P,
        B_FILMS_P,
        B_STUDENT_QUESTION_P,
        B_HINTS_P,
        B_BIBLIO_P,
        B_SESSIONS_P
    ]

    STATE_BEHS = []
    for beh in ALL_BEHS:
        if beh not in POINT_BEHS:
            STATE_BEHS.append(beh)

    POINT_STATE_BEH_MAP = {
        B_ASK_QUESTIONS_P: B_ASK_QUESTIONS_S,
        B_GIVE_QUESTIONS_P: B_GIVE_QUESTIONS_S,
        B_CHARTS_P: B_CHARTS_S,
        B_ORGANIZATION_P: B_ORGANIZATION_S,
        B_IMAGES_P: B_IMAGES_S,
        B_WEBSITE_P: B_WEBSITE_S,
        B_FILMS_P: B_FILMS_S,
        B_STUDENT_QUESTION_P: B_STUDENT_QUESTION_S,
        B_HINTS_P: B_HINTS_S,
        B_BIBLIO_P: B_BIBLIO_S,
        B_SESSIONS_P: B_SESSIONS_S
    }

    VISUAL_BEHS = {
        B_WRITING_SLIDES,
        B_WRITING_BOARD,
        B_CHARTS_S,
        B_CHARTS_P,
        B_IMAGES_S,
        B_IMAGES_P,
        B_MOVEMENT,
        B_STANDS,
        B_SESSIONS_S,
        B_SESSIONS_P,
        B_FILMS_S,
        B_FILMS_P,
        B_DEMONSTRATION,
        B_WEBSITE_S,
        B_WEBSITE_P
    }
    VISUAL_POINT_BEHS = []
    VISUAL_STATE_BEHS = []
    for beh in VISUAL_BEHS:
        if beh in POINT_BEHS:
            VISUAL_POINT_BEHS.append(beh)
        else:
            VISUAL_STATE_BEHS.append(beh)

    REL_STATE_BEH = [
        B_WRITING_SLIDES,
        B_CHARTS_S,
        B_IMAGES_S,
        B_WEBSITE_S,
        B_FILMS_S,
    ]

    REL_BEHS = [
        B_WRITING_SLIDES,
        B_CHARTS_S,
        B_CHARTS_P,
        B_IMAGES_S,
        B_IMAGES_P,
        B_WEBSITE_S,
        B_WEBSITE_P,
        B_FILMS_S,
        B_FILMS_P
    ]

    BEH_SHORT_LABELS = {
        B_WRITING_SLIDES: "writing",
        B_CHARTS_S: "S, charts",
        B_CHARTS_P: "P, charts",
        B_IMAGES_S: "S, images",
        B_IMAGES_P: "P, images",
        B_WEBSITE_S: "S, website",
        B_WEBSITE_P: "P, website",
        B_SESSIONS_S: "S, sessions",
        B_SESSIONS_P: "P, sessions",
        B_FILMS_S: "S, films",
        B_FILMS_P: "P, films"
    }

    SELECTED_POINT_BEHS = [
        B_FILMS_P,
        B_CHARTS_P,
        B_IMAGES_P,
        B_WEBSITE_P
    ]

    # Frames dataframe
    F_ID = "Id"
    F_VIDEO = "Video"
    F_EVENT_ID = "Event id"
    F_BEH_ID = "Beh id"
    F_RA_ID = "RA"
    F_TIME = "Time"
    F_COLL_EVENTS = "Collision events"
    F_BEHS = "Behs"
    F_LAYOUT = "Layout"
    F_LAYOUTS = "Layouts"
    F_SPLITS = "Splits"

    F_CONVERTERS = {
        F_COLL_EVENTS: eval,
        F_BEHS: eval,
        F_SPLITS: eval,
        F_LAYOUTS: eval,
    }

    # Views dataframe
    V_ID = "Id"
    V_EVENT_IDS = "Event ids"
    V_VIDEO = "Video"
    V_FRAME_NAME = "Frame name"
    V_VIEWS = "View names"
    V_BEH = "View behaviors"
    V_LAYOUTS = "View layouts"

    # https://stackoverflow.com/a/57373513
    V_CONVERTERS = {
        V_EVENT_IDS: eval,
        V_VIEWS: eval,
        V_BEH: eval,
        V_LAYOUTS: eval
    }

    # Sample dataframe
    S_FRAME_ID = "Frame id"
    S_SPLITS_NAMES = "Splits names"
    S_LABEL = "Label"

    S_CONVERTERS = {
        S_SPLITS_NAMES: eval,
    }

    # Layout labels
    LAYOUT_LABELS = {
        0: 'Not Answered',
        1: 'Camera',
        2: 'Screen',
        3: 'Left-Splited: Camera | Camera',
        4: 'Left-Splited: Camera | Screen',
        5: 'Left-Splited: Screen | Camera',
        6: 'Left-Splited: Screen | Screen',
        7: 'Middle-Splited: Camera | Camera',
        8: 'Middle-Splited: Camera | Screen',
        9: 'Middle-Splited: Screen | Camera',
        10: 'Middle-Splited: Screen | Screen'
    }
    SCREEN_LAYOUTS = [2, 4, 5, 6, 8, 9]

    SUBJECTS_DICT = subjects_dict = {
        "J1S1C11": [
            "PH1011-PHYSICS_20150922",
            "PH1011-PHYSICS_20151006",
            "PH1011-PHYSICS_20151013",
            "PH1011-PHYSICS_20151020"
        ],
        "J1S1C12": [
            "15S1-PH1012-LEC_20150812"
        ],
        "J1S1C13": [
            "PH1104-MECHANICS_20150817_PC1",
            "15S1-PH1104-LEC_20150820_PC1",
            "PH1104-MECHANICS_20150824_PC1",
            "PH1104-MECHANICS_20150827_PC1",
            "PH1104-MECHANICS_20150831_PC1"
        ],
        "J1S1C14": [
            "15S1-PH1105-LEC_20150817",
            "15S1-PH1105-LEC_20150821",
            "15S1-PH1105-LEC_20150824",
            "15S1-PH1105-LEC_20150828"
        ],
        "J1S1C5": [
            "15S1-MH1800-LEC_20150817",
            "15S1-MH1800-LEC_20150824",
            "15S1-MH1800-LEC_20150831"
        ],
        "J1S2C15": [
            "15S2-PH1106-LEC_20160113",
            "15S2-PH1106-LEC_20160120",
            "15S2-PH1106-LEC_20160127",
            "15S2-PH1106-LEC_20160203",
            "15S2-PH1106-LEC_20160210"
        ],
        "J1S2C4": [
            "15S2-MH1402-LEC_20160115",
            "15S2-MH1402-LEC_20160122",
            "15S2-MH1402-LEC_20160129",
            "15S2-MH1402-LEC_20160205",
            "15S2-MH1402-LEC_20160212"
        ],
        "J2S1C10": [
            "16S1-MH1812-LEC_20160808",
            "16S1-MH1812-LEC_20160811",
            "16S1-MH1812-LEC_20160815",
            "16S1-MH1812-LEC_20160818",
            "16S1-MH1812-LEC_20160822"
        ],
        "J2S1C13": [
            "16S1-PH1104-LEC_20161024",
            "16S1-PH1104-LEC_20161027",
            "16S1-PH1104-LEC_20161031",
            "16S1-PH1104-LEC_20161103",
            "16S1-PH1104-LEC_20161107",
            "16S1-PH1104-LEC_20161110"
        ],
        "J2S1C17": [
            "16S1-PH1801-LEC_20161013",
            "16S1-PH1801-LEC_20161020",
            "16S1-PH1801-LEC_20161027",
            "16S1-PH1801-LEC_20161103",
            "16S1-PH1801-LEC_20161110"
        ],
        "J2S2C3": [
            "16S2-MH1301-LEC_20170109",
            "16S2-MH1301-LEC_20170116",
            "16S2-MH1301-LEC_20170123",
            "16S2-MH1301-LEC_20170206",
            "16S2-MH1301-LEC_20170213"
        ],
        "J3S1C11": [
            "17S1-PH1011-LEC_20171010",
            "17S1-PH1011-LEC_20171017",
            "17S1-PH1011-LEC_20171024",
            "17S1-PH1011-LEC_20171031",
            "17S1-PH1011-LEC_20170912"
        ],
        "J3S1C17": [
            "17S1-PH1801-LEC_20170817",
            "17S1-PH1801-LEC_20170824",
            "17S1-PH1801-LEC_20170831",
            "17S1-PH1801-LEC_20170907",
            "17S1-PH1801-LEC_20170914",
            "17S1-PH1801-LEC_20170921"
        ],
        "J3S1C1": [
            "17S1-CM1021-LEC_20170823",
            "17S1-CM1021-LEC_20170906"
        ],
        "J3S1C2": [
            "17S1-MH1100-LEC_20170817",
            "17S1-MH1100-LEC_20171027",
            "17S1-MH1100-LEC_20170818",
            "17S1-MH1100-LEC_20170824",
            "17S1-MH1100-LEC_20170825",
            "17S1-MH1100-LEC_20170831",
            "17S1-MH1100-LEC_20171012",
            "17S1-MH1100-LEC_20171019",
            "17S1-MH1100-LEC_20171020",
            "17S1-MH1100-LEC_20171026"
        ],
        "J3S2C16": [
            "17S2-PH1107-CY1307-LEC_20180116",
            "17S2-PH1107-CY1307-LEC_20180123",
            "17S2-PH1107-CY1307-LEC_20180130",
            "17S2-PH1107-CY1307-LEC_20180206",
            "17S2-PH1107-CY1307-LEC_20180220"
        ],
        "J3S2C18": [
            "17S2-PH1802-LEC_20180316",
            "17S2-PH1802-LEC_20180324",
            "17S2-PH1802-LEC_20180406",
            "17S2-PH1802-LEC_20180413",
            "17S2-PH1802-LEC_20180420"
        ],
        "J3S2C3": [
            "17S2-MH1301-LEC_20180115",
            "17S2-MH1301-LEC_20180122",
            "17S2-MH1301-LEC_20180129",
            "17S2-MH1301-LEC_20180205",
            "17S2-MH1301-LEC_20180212",
            "17S2-MH1301-LEC_20180219"
        ],
        "J3S2C7": [
            "17S2-MH1803-LEC_20180116",
            "17S2-MH1803-LEC_20180117",
            "17S2-MH1803-LEC_20180123",
            "17S2-MH1803-LEC_20180124",
            "17S2-MH1803-LEC_20180130",
            "17S2-MH1803-LEC_20180131"
        ],
        "J3S2C8": [
            "17S2-MH1804-LEC_20180315",
            "17S2-MH1804-LEC_20180322",
            "17S2-MH1804-LEC_20180329",
            "17S2-MH1804-LEC_20180412",
            "17S2-MH1804-LEC_20180419"
        ],
        "J3S2C9": [
            "17S2-MH1811-LEC_20180205",
            "17S2-MH1811-LEC_20180207",
            "17S2-MH1811-LEC_20180212",
            "17S2-MH1811-LEC_20180214",
            "17S2-MH1811-LEC_20180219",
            "17S2-MH1811-LEC_20180221"
        ],
        "J4S1C2": [
            "18S1-MH1100-LEC_20180831",
            "18S1-MH1100-LEC_20181025",
            "18S1-MH1100-LEC_20181026",
            "18S1-MH1100-LEC_20180906",
            "18S1-MH1100-LEC_20180907",
            "18S1-MH1100-LEC_20180913",
            "18S1-MH1100-LEC_20180914",
            "18S1-MH1100-LEC_20181011",
            "18S1-MH1100-LEC_20181012",
            "18S1-MH1100-LEC_20181018",
            "18S1-MH1100-LEC_20181019"
        ],
        "J4S1C6": [
            "18S1-MH1802-LEC_20180820",
            "18S1-MH1802-LEC_20180827",
            "18S1-MH1802-LEC_20180829"
        ]
    }
    SUBJECTS_BY_VIDEO = {}
    for subj, videos in SUBJECTS_DICT.items():
        for video in videos:
            SUBJECTS_BY_VIDEO[video] = subj
