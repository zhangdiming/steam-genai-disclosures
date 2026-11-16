import re

F = re.IGNORECASE


def rx(*pats):
    return re.compile("|".join(f"(?:{p})" for p in pats), F)


AI = (r"(?:generative|gen ?ai|ai\b|a\.i\.|artificial intelligence|llms?\b|large language models?|language models?|gpt|chatgpt|"
      r"claude|gemini|copilot|codex|deepl|midjourney|stable diffusion|dall[-· ]?e|elevenlabs|suno|udio|firefly|leonardo|"
      r"nano banana|magnific|neural|machine learning|models?\b|tools?\b|photoshop'?s? generative|generative fill)")
NEG = rx(r"\b(no|not|never|none|nor|without|zero|neither)\b", r"n't\b", r"\bcannot\b", r"\bnon-ai\b")

RUNTIME = rx(r"\breal[- ]?time\b", r"\bduring (the )?(game ?play|play(ing)?|runtime|run[- ]time|a (run|session|playthrough))\b",
             r"\bwhile (the game is running|(you are |you're |players? (are|is) )?playing|you play|in[- ]game)\b", r"\bat runtime\b",
             r"\bon the fly\b", r"\blive[- ]?generat\w*", r"\bgenerated live\b", r"\blive generation\b",
             r"\bdynamically (generat|simulat|creat|respond|adapt|react|produc)", r"\bgenerated dynamically\b",
             r"\b(reacts?|responds?|adapts?) to (your|the players?'?s?|player) ", r"\bin response to (the )?(players?|your|user)",
             r"\bchat(s|ting)? with (the )?(players?|you|characters?|npcs?)\b", r"\b(talk|speak) (freely )?(to|with)\b", r"\bconvers(e|ation)",
             r"\b(type|enter|write|ask)\w* (your own |any |a |custom )?(prompts?|questions?|messages?)\b",
             r"\b(based on|from) (the )?(your|players?'?s?|user'?s?) (own )?(text |voice |image )?(inputs?|prompts?|descriptions?|requests?|messages?)\b",
             r"\bplayers? (can|may|are able to|will be able to|get to) (\w+ ){0,3}(generat|creat|design|make)\w*[^.;]{0,50}\b(ai|prompt|text input|descri)",
             r"\b(on[- ]device|locally on|runs? locally|local (llm|model|ai))\b", r"\b(llm|ai)[- ](driven|powered|controlled) (npc|characters?|companions?|dialogues?|conversations?|narrat\w*|story|game ?masters?|pets?|systems?|opponents?)",
             r"\bgenerates? (unique |new |custom )?(dialogues?|responses?|conversations?|stories|quests?|answers?)\b",
             r"\bno pre-?written\b", r"\beach (playthrough|run|session) (is )?(unique|different)\b", r"\bno two playthroughs\b")
PLAYER = rx(r"\b(type|enter|write|ask)\w* (your own |any |a |custom )?(prompts?|questions?|messages?|text)\b",
            r"\b(based on|from|using) (the )?(your|players?'?s?|user'?s?) (own )?(text |voice |image )?(inputs?|prompts?|descriptions?|requests?|messages?)\b",
            r"\bplayers? (can|may|are able to|will be able to|get to)\b", r"\byou can (generate|create|design|make|talk|chat|ask)\b",
            r"\bchat(s|ting)? with\b", r"\b(talk|speak) (freely )?(to|with)\b", r"\bconvers(e|ation)", r"\bfree[- ]?(form|text)\b",
            r"\buser[- ]generated\b", r"\bjudg\w* (the |your |whether )", r"\bprompts?\b")
MODERATION_ONLY = rx(r"\bmoderat\w*|\bfilter\w* (out )?(inappropriate|chat)")
GENERATE = rx(r"\bgenerat\w*|\bcreat\w*|\bproduc\w*")
PREGEN = rx(r"\b(all|everything|any|these|this|such)\b[^.;]{0,60}\b(pre[- ]?generated|pre[- ]?rendered|pre[- ]?made|created before (release|launch)|generated before|made before|prepared in advance|included as (fixed|finished)|fixed (assets|content)|static)")

MKT = rx(r"\bstore[- ]?(page|front)?s?'?\s*(capsules?|art\w*|images?|imagery|visuals?|assets?|materials?|graphics|text|copy|descriptions?|screenshots?|banners?|listings?|music)?",
         r"\bsteam (page|store|description|capsules?|trading cards?|assets?|marketing)\b", r"\bcapsules?( art\w*| images?)?\b",
         r"\b(key|library|header|hero|cover|promotional|promo|marketing|advertis\w*|social[- ]media|trailer)( page)?\s*(art\w*|images?|imagery|assets?|materials?|text|copy|descriptions?|videos?|music|graphics|visuals?|pictures?|posts?)?",
         r"\btrailers?\b", r"\btrading cards?\b", r"\bemoticons?\b", r"\bprofile backgrounds?\b", r"\bthumbnails?\b",
         r"\b(game )?logos?\b", r"\bbanners?\b", r"\bachievement icons? for steam\b")
CONTENT = rx(r"\b(art|artwork|arts|assets?|images?|illustrations?|sprites?|textures?|3d|models?|graphics?|visuals?|icons?|ui|interface|"
             r"backgrounds?|portraits?|animations?|videos?|cutscenes?|scenes?|music|songs?|soundtracks?|bgm|tracks?|audio|sounds?|sfx|"
             r"voices?|voice[- ]?overs?|narration|texts?|dialogues?|story|stories|writing|lore|descriptions?|translations?|translat\w*|"
             r"locali[sz]\w*|languages?|code|coding|programming|scripts?|logic|bugs?|levels?|maps?|items?|cards?|characters?|content|"
             r"materials?|development|in-game|environments?|effects|props|posters|menus?|quests?|names?)\b")

EDITV = (r"(?:edit|refin|retouch|modif|adjust|rework|tweak|polish|paint(?:ed)? over|overpaint|correct|fix|clean|alter|revis|"
         r"rewrit|re-?written|mix|post[- ]?process|process|touch(?:ed)? up|combin|customi[sz]|improv|enhanc|transform|iterat|"
         r"optimi[sz]|redr[ae]w|redo|redone|composit|crop|recolou?r|adapt|repair|author|optimiz)\w*")
REVV = r"(?:review|check|curat|select|approv|verif|vett|inspect|screen|supervis|chos|pick|integrat|finali[sz]|validat|test|proofread)\w*"
AI_AFTER = re.compile(r"^\W*(?:\w+\W+){0,5}?(?:with|using|via|through|by)\s+(?:the\s+)?(?:help\s+of\s+|assistance\s+of\s+|aid\s+of\s+|support\s+of\s+)?"
                      r"(?:an?\s+)?(?:open-source\s+)?" + AI, F)
HUMAN_AFTER = re.compile(r"^\W*(?:\w+\W+){0,6}?by\s+(?:the\s+|our\s+|a\s+|an\s+|my\s+)?(?:hand|humans?|developers?|devs?|team|artists?|"
                         r"designers?|writers?|translators?|creators?|staff|studio|me|us|myself|ourselves|people|person|solo|art|"
                         r"human|professional|in-house|internal|real)", F)
AI_TO_BEFORE = re.compile(AI + r"[^.;]{0,40}\b(?:to|for)\s+(?:\w+\s+){0,2}$", F)
PASSIVE_LEAD = r"(?:were|was|been|be|is|are|being|then|and|subsequently|later|further|afterwards|manually|carefully|heavily|extensively|lightly|partially|thoroughly|personally|individually|always|all|finally|also|hand-|,)"
EDIT_PASSIVE = re.compile(r"\b" + PASSIVE_LEAD + r"\s+(?:(?:\w+ly|further|then|and|also|all|carefully|,)\s+){0,4}(" + EDITV + r")\b", F)
REV_PASSIVE = re.compile(r"\b" + PASSIVE_LEAD + r"\s+(?:(?:\w+ly|further|then|and|also|all|carefully|,)\s+){0,4}(" + REVV + r")\b", F)
HUMAN_SUBJ = re.compile(r"\b(?:we|i|our (?:team|artists?|designers?|developers?|art team|writers?)|the (?:team|developers?|artists?|designers?|"
                        r"studio|art director)|(?:a |our )?human (?:artists?|creators?|designers?|translators?|editors?))\s+(?:\w+\s+){0,3}?("
                        + EDITV + "|" + REVV + r")\b", F)
EDIT_NOUN = rx(r"\b(human|manual|hand|artist) (edit\w*|refine\w*|adjust\w*|post-?process\w*|touch-?ups?|correct\w*|rework\w*|retouch\w*|"
               r"clean-?up|modification\w*|polish\w*|work|optimi[sz]ation|processing)\b", r"\bsecondary (optimi[sz]ation|processing|editing)\b",
               r"\bmanually (\w+ )?(" + EDITV + r")")
REV_NOUN = rx(r"\b(human|manual|developer) (review|oversight|supervision|curation|selection|approval|checks?|verification|quality control)\b",
              r"\bunder (the )?(human|our|the developer'?s?|the team'?s?|direct) (creative )?(direction|supervision|oversight|control)\b",
              r"\bquality (control|checks?|assurance)\b", r"\bmanually (reviewed|checked|selected|curated|verified|approved)\b")
FINAL_ROLE = rx(r"\b(as|for|of)\s+(an?\s+|the\s+)?(visual\s+|rough\s+|initial\s+|early\s+|artistic\s+|creative\s+)?(references?|inspirations?|"
                r"ideation|brainstorm\w*|mood ?boards?|ideas?)\b",
                r"\breferences? (for|materials?|images?)\b", r"\b(concept|conceptual)( art| arts| design| drafts?| images?| sketch\w*| ideas?| stage)?\b",
                r"\bart direction\b", r"\b(explore|validate|exploring|validating) (various )?(artistic )?(styles?|directions?|ideas)\b",
                r"\bmock-?ups?\b", r"\binspir\w+\b")
FINAL_STMT = rx(r"\b(final|finished|shipped|all (of the |the )?(in-game|gameplay|final|core|key|playable)|every (key |single )?(asset|character|visual|image|illustration|line))\b"
                r"[^.;]{0,90}\b(made|created|drawn|painted|crafted|produced|done|written|authored|designed|redrawn|re-?drawn|rebuilt|hand[- ]?\w+|in the hands|human[- ]made|by hand|manually)\b",
                r"\b(done|made|created|written|drawn|translated) by (humans?|hand|our (team|artists?)|me|us|myself|the (developers?|team|artists?))[^.;]{0,25}\b(but |while |though )?(aided|assisted|helped|supported)\b",
                r"\breplaced (with|by) (human|hand|original|our own|custom|professional)", r"\bnot a game made by ai\b")
FINAL_EXCL = rx(r"\b(remaining|other|rest|everything else|all else)\b", r"\b(will|to) be\b", r"\bplan\w*\b", r"\bbeing (re\w+|replaced)\b",
                r"\b(gradually|eventually|in the future|as development progresses)\b")

AI_ON_HUMAN = [
    re.compile(AI + r"[^.;]{0,40}\b(?:to|for|in)\b[^.;]{0,25}\b(modif|edit|polish|enhanc|upscal|up-?convert|upconvert|refin|optimi[sz]|adjust|"
               r"clean|correct|improv|revis|proofread|debugg?|review|beautif|process|sharpen|redr|restor|elevat|fix|detail|retouch|denois|vari)\w*", F),
    re.compile(r"\b(modif|edit|polish|enhanc|upscal|up-?convert|refin|optimi[sz]|adjust|clean|correct|improv|revis|proofread|debugg?|beautif|"
               r"process|sharpen|redr|restor|elevat|fix|detail|retouch)\w*\s+(?:\w+\s+){0,3}?(?:with|using|via|through|by)\s+(?:the\s+)?"
               r"(?:help\s+of\s+|assistance\s+of\s+|aid\s+of\s+)?(?:an?\s+)?(?:open-source\s+)?" + AI, F),
    rx(r"\bcreat\w* or (modif|edit|refin|improv|revis|enhanc)\w*", r"\bcreation and (improvement|revision|enhancement|refinement|modification)\b",
       r"\b(code review|bug ?-?fix\w*|debugg\w*|proofread\w*|spell(ing)? (check\w*|errors?)|grammar\w*|upscal\w*|up-?convert\w*)",
       r"\bbased on (our|my|the original|the existing|existing|pre-?existing|original|photographs?|photos|online drafts|(our|my) own)\b",
       r"\bfrom (the original|our own|my own|existing|the existing) (game'?s? )?(designs?|art|artwork|assets|drafts|sketches|photos|characters?)\b",
       r"\b(our|my) own (custom )?(art|artwork|characters?|photos?|drawings?|sketches|designs?|line drawings?)\b",
       r"\bhand[- ]drawn (then|and then) [^.;]{0,20}\b(ai|with)\b", r"\b(made|drawn|created) (manually|by hand)[^.;]{0,60}\b(ai|neural)\b"),
]

D = {
    "d_visual": rx(r"\b(art|artwork|arts|images?|imagery|illustrations?|sprites?|textures?|3d|models?|graphics?|visuals?|pictures?|portraits?|"
                   r"backgrounds?|icons?|ui|interface|animations?|animated|cutscenes?|videos?|concept art|character designs?|tiles?|cgs?|"
                   r"paintings?|drawings?|upscal\w*|photos?|skybox|environments?|capsules?|posters?|logos?|banners?|key art|thumbnails?|"
                   r"visual effects|vfx|renders?|sketch\w*|frames?|scenes?)\b"),
    "d_music": rx(r"\b(music|soundtracks?|songs?|bgm|ost|melod\w*|tracks?)\b"),
    "d_sfx": rx(r"\b(sound effects?|sfx|sounds?|audio|ambien\w*)\b"),
    "d_voice": rx(r"\b(voices?|voice[- ]?overs?|voice acting|voiced|vo|narrat(ion|or|ed)|text[- ]to[- ]speech|tts|speech|dubbing|voice lines?)\b"),
    "d_text": rx(r"\b(writing|written content|story|stories|narrative|dialogues?|dialog|lore|flavou?r text|in-game text|game'?s? text|"
                 r"item descriptions?|descriptions|quests?|names?|poems?|lyrics|captions?|prose|questions and answers|trivia|events|"
                 r"textual|texts?|wording|plot)\b"),
    "d_localization": rx(r"\b(translat\w*|locali[sz]\w*|languages?)\b"),
    "d_code": rx(r"\b(code|coding|programming|programm\w*|scripts?|scripting|shaders?|debug\w*|logic|software|bugs?|bug ?fix\w*|codebase)\b"),
    "d_other": rx(r"\b(ideas?|ideation|brainstorm\w*|design documents?|level design|game design|balanc\w*|playtest\w*|concepts?|conceptual|"
                  r"planning|research|prototyp\w*|mock-?ups?|drafts?|references?|inspiration|specifications?)\b"),
}
TEXT_EXCL = rx(r"\b(locali[sz]ation|translat\w*|store|marketing|steam page|promotional)\s+(\w+\s+)?(texts?|copy|descriptions?)\b",
               r"\btexts? (into|to) (\w+ )?languages\b", r"\btranslat\w* (\w+ ){0,3}texts?\b", r"\binterface texts?\b")

SCOPE_MINOR = rx(r"\b(some|a few|few|minor|small (portion|number|amount|part)|limited|one or two|a couple|several|certain|partial(ly)?|"
                 r"a portion|portions? of|selected|select|occasional(ly)?|small|minimal|parts? of|a number of|a handful)\b",
                 r"\bonly (used )?(for|in|to)\b")
QUANT = r"(?:all|most|majority of|entire(?:ly)?|much of|many of|the bulk of|nearly all|almost all|every)"
SCOPE_EXT = rx(QUANT + r"\s+(?:of\s+)?(?:the\s+)?(?:game'?s?\s+|in-game\s+|other\s+)?(?:\w+\s+){0,1}(?:art\w*|visual\w*|assets?|graphics|images?|illustrations?|music|bgm|songs?|tracks?|sounds?|audio|voices?|voice\w*|texts?|dialogues?|code|content|localizations?|translations?|languages|loading screens?|backgrounds?|characters?|3d models?|models|icons?|materials?|cards?)",
               r"\b(fully|entirely|completely|mostly|primarily|largely|heavily|extensively)\s+(?:\w+\s+){0,2}(generat|produc|creat|made|develop|built)\w*\s+(with|by|using|through)",
               r"\b(heavy|extensive) use of\b", r"\bprimary production tool\b", r"\ba lot of\b", r"\bmany of\b", r"\bmuch of\b",
               r"\bmost (of )?(the )?\w+", r"\b(are|were|is) all (generated|created|made)\b", r"\bevery character\b")
SCOPE_EXT_EXCL = rx(r"\b(by hand|human|hand[- ]\w+|manually|our (team|artists?)|ourselves|in-house)\b", r"\bai[- ]generated (content|assets?|material)\b[^.;]{0,30}\b(was|were|is|are|has|have)\b")
PLACEHOLDER = rx(r"\bplace ?holders?\b", r"\btemporar(y|ily)\b", r"\bwill be replaced\b", r"\bplan(s|ning)? to replace\b", r"\bfor now\b",
                 r"\b(being|be) (gradually )?(replaced|redesigned|redrawn|reworked)\b")
TOOLS = rx(r"\b(midjourney|stable diffusion|sdxl|\bsd\b|dall[-· ]?e|chat ?gpt|gpt[- ]?\d|gpt\b|openai|claude|anthropic|gemini|bard|copilot|"
           r"elevenlabs|eleven labs|11\.ai|suno|udio|firefly|adobe|leonardo|meshy|tripo|codex|cursor|deepseek|qwen|llama|mistral|grok|flux|"
           r"runway|sora|kling|veo|ideogram|novelai|niji|comfyui|automatic1111|deepl|whisper|kimi|doubao|nano banana|imagen|rodin|hunyuan|"
           r"krea|pixverse|luma|soundraw|seedance|lyria|scenario|magnific|aiva|artlist|photoshop)\b")
PROVENANCE = rx(r"\b(licens\w*|ethical\w*|commercially safe|trained (on|with)|own (original )?(art|artwork|data|dataset|style|assets)|in-house (model|data)|"
                r"copyright\w*|royalty[- ]free|public domain|consent|cc0|likeness\w*|infring\w*|impersonat\w*)\b")
SMALL_TEAM = rx(r"\b(solo|one[- ](person|man|woman)|single developer|small (indie )?(team|studio)|tiny team|lone developer|indie developers?|"
                r"independent (developer|team|studio)|limited (budget|resources|means)|small budget|only developer|by myself|on my own|just me|"
                r"two-person|(two|three|four|five|six)[- ](person|people) team|do not have the means|out of reach)\b")
MIXED = rx(r"\b(unmodified|unedited|as[- ]is|without (any )?(modification|editing|changes))\b")
NON_GENAI = rx(r"\b(enemy|enemies|opponent|bot|npc) (ai|behaviou?r)\b", r"\bai (opponents?|enemies|bots?|players?|difficulty)\b",
               r"\bchess engines?\b", r"\bpath ?finding\b", r"\bbehaviou?r trees?\b", r"\bstate machines?\b")
GENAI = rx(r"\bgenerat\w*", r"\bai[- ]", r"\b(assist|help|support|tool|creat)\w*", r"\b(midjourney|stable diffusion|chat ?gpt|gpt|llm|suno|elevenlabs)\b")

GR = {
    "gr_filter": rx(r"\bfilter\w*\b", r"\bmoderat\w*\b", r"\bblock ?lists?\b", r"\bblock(s|ing|ed)?\b", r"\bsafety\b", r"\bcensor\w*\b",
                    r"\bprofan\w*\b", r"\binappropriate\b", r"\bharmful\b", r"\boffensive\b", r"\billegal\b", r"\bnsfw\b", r"\bexplicit\b",
                    r"\bguard ?rails?\b", r"\bsafeguards?\b"),
    "gr_constrained": rx(r"\bconstrain\w*\b", r"\brestrict\w*\b", r"\blimit(ed|s)? (to|in)\b", r"\bwithin (the )?(existing |game'?s? )?(context|world|setting|lore|rules|scope|universe)\b",
                         r"\bsystem prompts?\b", r"\bpre-?defined\b", r"\bpreset\b", r"\bcannot (generate|produce|create|say)\b",
                         r"\bcan'?t (generate|produce|create)\b", r"\bcurated\b", r"\bguided\b", r"\bboundaries\b", r"\bstays? in character\b",
                         r"\btemplates?\b", r"\bfixed (set|list|options)\b", r"\bonly (generate|respond|talk|discuss)\w*"),
    "gr_local": rx(r"\blocal(ly)?\b", r"\bon[- ]device\b", r"\boffline\b", r"\b(data|information|inputs?|conversations?)[^.;]{0,30}\b(not |never )?(sent|stored|collected|shared|transmitted|saved)\b",
                   r"\bprivacy\b", r"\bprivate\b", r"\bdoes not (connect|send|collect|store)\b"),
    "gr_human": rx(r"\b(human|manual(ly)?) (moderat\w*|review\w*|oversight|monitor\w*)\b",
                   r"\b(our|the) (team|developers?|staff|moderators?) (will |actively |regularly |continuously )?(monitor|review|check|audit)\w*",
                   r"\bmonitor(ed|ing)?\b", r"\blogs?\b", r"\baudit\w*\b"),
    "gr_report": rx(r"\breport\w*\b", r"\bfeedback\b", r"\bcontact (us|the developer)\b", r"\boverlay\b", r"\bflag\w*\b"),
    "gr_optout": rx(r"\b(turn(ed)? off|switch(ed)? off|disabl\w*|toggle\w*|opt[- ]?(in|out)|optional(ly)?|can choose|choose (not )?to|in the settings)\b"),
    "gr_provider": rx(r"\b(openai|anthropic|google|microsoft|provider|vendor|third[- ]party)('?s)? (moderation|safety|usage|content) (polic\w*|api|filters?|guidelines|systems?)\b",
                      r"\b(usage|content) polic(y|ies) of\b", r"\bterms of (service|use)\b"),
}


def clauses(text):
    t = re.sub(r"[•·・●▪■◆①②③④⑤]|^\s*[-*]\s+", "\n", text or "", flags=re.M)
    parts = re.split(r"(?<=[.!?])\s+|\n+|;\s*|\s+—\s+", t)
    return [p.strip() for p in parts if p and p.strip()]


def _human_edit(s):
    for m in list(EDIT_PASSIVE.finditer(s)) + list(HUMAN_SUBJ.finditer(s)):
        verb_end = m.end()
        tail = s[verb_end:]
        head = s[:m.start(1)] if m.lastindex else s[:m.start()]
        if AI_AFTER.search(tail) and not HUMAN_AFTER.search(tail):
            continue
        if AI_TO_BEFORE.search(head):
            continue
        return True
    return False


def _human_review(s):
    for m in list(REV_PASSIVE.finditer(s)) + list(HUMAN_SUBJ.finditer(s)):
        word = m.group(1).lower()
        if not re.match(REVV, word, F):
            continue
        tail = s[m.end():]
        head = s[:m.start(1)]
        if AI_AFTER.search(tail) and not HUMAN_AFTER.search(tail):
            continue
        if AI_TO_BEFORE.search(head):
            continue
        if word.startswith("select") and not re.search(r"\b(were|was|be|been|are|is|then|and|carefully|manually)\s+(\w+\s+){0,2}select", s, F):
            continue
        return True
    return False


def code_disclosure(text):
    t = text or ""
    cl = clauses(t)
    pos = [s for s in cl if not NEG.search(s)]
    neg = [s for s in cl if NEG.search(s)]
    c = {}

    rt = [s for s in pos if RUNTIME.search(s) and not (MODERATION_ONLY.search(s) and not GENERATE.search(s))]
    c["live_system"] = int(bool(rt))
    c["live_player"] = int(any(PLAYER.search(s) for s in rt))
    c["explicit_no_live"] = int(any(RUNTIME.search(s) or re.search(r"\b(live|connect\w*|external|online|services?|embedded|runs? while)\b", s, F) for s in neg)
                                or bool(PREGEN.search(t)))
    marketing = any(MKT.search(s) for s in pos)
    c["marketing"] = int(marketing)
    nonrt = [s for s in pos if s not in rt]
    ingame = any(CONTENT.search(MKT.sub(" ", s)) for s in nonrt)
    c["non_genai"] = int(bool(NON_GENAI.search(t)) and not GENAI.search(t))
    live = c["live_system"] or c["live_player"]
    c["marketing_only"] = int(marketing and not ingame and not live)
    c["pre"] = int(not c["marketing_only"] and not c["non_genai"] and (ingame or not live))

    a = 0
    if any(_human_review(s) for s in pos) or any(REV_NOUN.search(s) for s in pos):
        a = 1
    if any(_human_edit(s) for s in pos) or any(EDIT_NOUN.search(s) for s in pos):
        a = 2
    if any(FINAL_ROLE.search(s) for s in pos) or any(FINAL_STMT.search(s) and not FINAL_EXCL.search(s) for s in cl):
        a = 3
    c["authority"] = a
    c["mixed_oversight"] = int(bool(MIXED.search(t)))
    c["ai_on_human"] = int(any(p.search(s) for s in pos for p in AI_ON_HUMAN))

    for k, p in D.items():
        c[k] = int(any(p.search(s) for s in pos))
    c["d_text"] = int(any(D["d_text"].search(TEXT_EXCL.sub(" ", s)) for s in pos))
    c["d_marketing"] = c["marketing"]
    c["scope_minor"] = int(any(SCOPE_MINOR.search(s) for s in pos))
    c["scope_extensive"] = int(any(SCOPE_EXT.search(s) and not SCOPE_EXT_EXCL.search(s) for s in pos))
    c["placeholder"] = int(bool(PLACEHOLDER.search(t)))
    c["tools_named"] = int(bool(TOOLS.search(t)))
    c["provenance_claim"] = int(bool(PROVENANCE.search(t)))
    c["small_team"] = int(bool(SMALL_TEAM.search(t)))
    for k, p in GR.items():
        c[k] = int(bool(live and p.search(t)))
    c["gr_none"] = int(bool(live) and not any(c[k] for k in GR))
    c["n_words"] = len(t.split())
    return c
