import re

F = re.IGNORECASE
STRONG = re.compile(r"|".join([
    r"\bai[- ]?(generated|gen|made|created|written|drawn|art|arts|artwork|images?|pictures?|assets?|graphics|slop|voices?|voice[- ]?(over|acting|actors?)|music|songs?|soundtrack|content|translat\w*|tools?|disclosure|garbage|trash|crap|junk|shit|stuff|bullshit|nonsense|nightmare|portraits?|sprites?|textures?|models?|illustrations?|backgrounds?|dialogue|text|writing|story|narrat\w*|cover|capsule|thumbnail|key ?art|trailer|filler|look(ing)?|style)\b",
    r"\b(generative|gen) ?ai\b", r"\bgenai\b", r"\bmidjourney\b", r"\bstable diffusion\b", r"\bdall[-· ]?e\b",
    r"\bchat ?gpt\b", r"\bllms?\b", r"\blarge language model", r"\belevenlabs\b", r"\bsuno\b",
    r"\b(made|created|generated|written|drawn|done|produced|built|voiced|translated) (with|by|using|through) (an? )?(generative )?ai\b",
    r"\b(uses?|using|used|use of|usage of|relies on|reliance on) (generative )?ai\b", r"\bai (was|is|were) used\b",
    r"\bmachine[- ]translat\w*\b", r"\bgoogle translat\w*\b",
]), F)
GAME_AI = re.compile(r"|".join([
    r"\b(enemy|enemies|opponent|opponents|bot|bots|npc|npcs|companion|ally|allies|squad|teammate|teammates|unit|units|cpu|computer) ai\b",
    r"\bai (enemies|enemy|opponents?|bots?|difficulty|players?|teammates?|squad|units?|drivers?|behaviou?r|path ?finding|logic|controlled|director|routines?)\b",
    r"\bthe ai (is|was|are|were|seems|feels|can|cannot|can't|doesn't|does|will|keeps?|always|never|just)\b",
    r"\b(dumb|stupid|smart|braindead|brain-dead|cheating|aggressive|passive|good|bad|terrible|decent|clever|challenging) ai\b",
    r"\b(hostile|friendly|enemy|guard|guards|police|traffic|racing|rival|zombie|monster|boss|combat|tactical|squad|team) ai\b",
    r"\bai (is|was|are|were|seems?|feels?) (so |very |really |pretty |kinda |too |extremely |incredibly )?"
    r"(braindead|brain-dead|dumb|stupid|basic|predictable|easy|hard|challenging|aggressive|passive|smart|clever|"
    r"cheap|unfair|broken|buggy|janky|competent|incompetent|relentless)\b",
    r"\b(bugs?|glitch\w*|pathing|difficulty) (or|and|/) ai\b|\bai (issues|bugs?|glitch\w*)\b",
]), F)
AI_TOKEN = re.compile(r"\bai\b|\ba\.i\.|\bartificial intelligence\b", F)
CRITICAL = re.compile(r"|".join([
    r"\bslop\w*\b", r"\blazy\b", r"\blaziness\b", r"\bcheap\w*\b", r"\bgarbage\b", r"\btrash\b", r"\bjunk\b", r"\bcrap\w*\b", r"\bshit\w*\b",
    r"\bsoulless\b", r"\bno soul\b", r"\blifeless\b", r"\bgeneric\b", r"\brefund\w*\b", r"\bavoid\b", r"\bdisappoint\w*\b", r"\bshame\w*\b",
    r"\bstolen\b", r"\bsteal\w*\b", r"\btheft\b", r"\bthie(f|ves|ving)\b", r"\bplagiar\w*\b", r"\bscam\w*\b", r"\bshovelware\b", r"\basset ?flip\w*\b",
    r"\b(no|low|zero|minimal|little) effort\b", r"\bhate\b", r"\bdisgust\w*\b", r"\bboycott\w*\b", r"\bwon'?t support\b", r"\bnot support\w*\b",
    r"\bugly\b", r"\bobvious(ly)?\b", r"\bcan (clearly |easily )?tell\b", r"\buncanny\b", r"\bawful\b", r"\bterrible\b", r"\bhorrible\b",
    r"\bwaste\b", r"\bdo not buy\b", r"\bdon'?t buy\b", r"\bnot worth\b", r"\bunfortunately\b", r"\bsadly\b", r"\bpathetic\b",
    r"\bdisrespect\w*\b", r"\binsult\w*\b", r"\bunethical\b", r"\bnot (a |an )?fan\b", r"\bturn(ed|s)? (me )?off\b", r"\bdeal ?breaker\b",
    r"\bred flag\b", r"\bnope\b", r"\bfuck\w*\b", r"\bawful\b", r"\bcringe\b", r"\bweird\b", r"\bjarring\b", r"\bjank\w*\b", r"\bmeh\b",
    r"\bdislike\w*\b", r"\bbother\w*\b", r"\bannoy\w*\b", r"\bpoor\b", r"\bbad\b", r"\bworse\b", r"\bworst\b", r"\bsucks?\b",
]), F)
ACCUSE = re.compile(r"|".join([
    r"\bundisclosed\b", r"\b(not|never|wasn'?t|isn'?t|weren'?t|aren'?t) (been )?(properly |clearly )?disclosed\b",
    r"\b(didn'?t|did not|doesn'?t|does not|don'?t|do not|never|failed to|fail to|refuse[sd]? to|want to) (even )?disclose\b",
    r"\bwithout (any |proper |a )?(disclos\w*|telling|saying|mentioning|warning|admitting)\b",
    r"\b(no|lack of|missing|zero|without an?) (ai |a\.i\. |gen ?ai |generative ai )?(disclosure|disclaimer|label|tag|warning|notice)\b",
    r"\bnot (noted|mentioned|listed|stated|labell?ed|tagged|shown|said|marked) (in|on|anywhere)\b",
    r"\bno ?where (on|in) the (store|steam|page|description)", r"\b(called out|caught)\b",
    r"\b(lie[sd]?|lying|liar|dishonest\w*|deceiv\w*|deceptive\w*|misleading|hid(e|ing|den)?|conceal\w*)\b[^.!?]{0,30}"
    r"\b(about|that|the fact|it|ai|usage|use)\b",
    r"\b(disclosure|disclaimer)\b[^.!?]{0,40}\b(lie|false|inaccurate|understat\w*|incomplete|misleading|only|downplay\w*)\b",
    r"\b(beyond|more than) (what|the items?|the (ai )?disclosure)[^.!?]{0,30}(disclosed|reported|stated|says)",
    r"\b(added|adding|update[sd]?)\b[^.!?]{0,40}\b(disclosure|ai (tag|label|notice|warning|box))",
]), F)

DISCREF = re.compile(r"|".join([
    r"\bdisclos\w*", r"\bdisclaimer\b",
    r"\b(store|steam|shop) ?page\b[^.!?]{0,40}\b(say|says|said|state[sd]?|mention\w*|note[sd]?|list\w*|admit\w*|warn\w*|clearly|"
    r"openly)\b",
    r"\b(ai|a\.i\.|gen ?ai|generative ai) (content )?(label|tag|notice|warning|section|disclaimer|statement)\b",
    r"\b(devs?|developers?|they|studio|creators?) (openly |clearly |even |do |did )?(say|says|said|state[sd]?|claim\w*|admit\w*|"
    r"mention\w*|note[sd]?|disclose\w*|list\w*|were upfront|are upfront)\b[^.!?]{0,40}\b(ai|a\.i\.)\b",
    r"\baccording to the (store|steam|dev\w*|page|description)\b", r"\b(upfront|transparent|honest|open) about\b",
    r"\bin the (store |steam )?(page|description)\b",
]), F)
OVERREF = re.compile(r"|".join([
    r"\bhuman[- ](reviewed|edited|made|touched|refined|curated|checked|crafted)\b",
    r"\b(touched up|edited|reviewed|refined|polished|reworked|fixed|redrawn|cleaned up|curated|checked) by (a |an |the |real |"
    r"human )*(humans?|artists?|team|devs?|developers?|people|person)\b",
    r"\b(humans?|artists?) (then )?(edited|touched|reviewed|refined|fixed|reworked|redrew|polished|cleaned)\b",
    r"\bplaceholders?\b", r"\b(only|just) (used )?(as|for) (a )?(reference|inspiration|base|starting point)\b",
]), F)


def _sents(t):
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", t or "") if s.strip()]


def genai_sentences(text):
    out = []
    for s in _sents(text):
        if STRONG.search(s):
            out.append(s)
        elif AI_TOKEN.search(s) and not GAME_AI.search(s):
            out.append(s)
    return out


def genai_mention(text):
    return int(bool(genai_sentences(text)))


def genai_critical(text, recommended=None):
    sents = genai_sentences(text)
    if not sents:
        return 0
    return int(any(CRITICAL.search(s) for s in sents) or recommended == 0)


def genai_accusation(text):
    return int(any(ACCUSE.search(s) for s in genai_sentences(text)))


def genai_disclosure_ref(text):
    return int(any(DISCREF.search(s) for s in genai_sentences(text)))


def genai_oversight_ref(text):
    return int(any(OVERREF.search(s) for s in genai_sentences(text)))
