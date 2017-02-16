# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import os.path

from sklearn.externals.joblib import dump, load

CUSTOM_STOP_WORDS = ['com']

# List of common english first names
# source http://names.mongabay.com/male_names.htm
# source http://names.mongabay.com/female_names.htm
COMMON_FIRST_NAMES = [ "james","john","robert","michael","william","david",
"richard","charles","joseph","thomas","christopher","daniel","paul","mark",
"donald","george","kenneth","steven","edward","brian","ronald","anthony",
"kevin","jason","matthew","gary","timothy","jose","larry","jeffrey","frank",
"scott","eric","stephen","andrew","raymond","gregory","joshua","jerry","dennis",
"walter","patrick","peter","harold","douglas","henry","carl","arthur","ryan",
"roger","joe","juan","jack","albert","jonathan","justin","terry","gerald","keith",
"samuel","willie","ralph","lawrence","nicholas","roy","benjamin","bruce","brandon",
"adam","harry","fred","wayne","billy","steve","louis","jeremy","aaron","randy",
"howard","eugene","carlos","russell","bobby","victor","martin","ernest","phillip",
"todd","jesse","craig","alan","shawn","clarence","sean","philip","chris","johnny",
"earl","jimmy","antonio","danny","bryan","tony","luis","mike","stanley","leonard",
"nathan","dale","manuel","rodney","curtis","norman","allen","marvin","vincent",
"glenn","jeffery","travis","jeff","chad","jacob","lee","melvin","alfred","kyle",
"francis","bradley","jesus","herbert","frederick","ray","joel","edwin","don",
"eddie","ricky","troy","randall","barry","alexander","bernard","mario","leroy",
"francisco","marcus","micheal","theodore","clifford","miguel","oscar","jay","jim",
"tom","calvin","alex","jon","ronnie","bill","lloyd","tommy","leon","derek","warren",
"darrell","jerome","floyd","leo","alvin","tim","wesley","gordon","dean","greg",
"jorge","dustin","pedro","derrick","dan","lewis","zachary","corey","herman",
"maurice","vernon","roberto","clyde","glen","hector","shane","ricardo","sam",
"rick","lester","brent","ramon","charlie","tyler","gilbert","gene","marc",
"reginald","ruben","brett","angel","nathaniel","rafael","leslie","edgar","milton",
"raul","ben","chester","cecil","duane","franklin","andre","elmer","brad","gabriel",
"ron","mitchell","roland","arnold","harvey","jared","adrian","karl","cory","claude",
"erik","darryl","jamie","neil","jessie","christian","javier","fernando","clinton",
"ted","mathew","tyrone","darren","lonnie","lance","cody","julio","kelly","kurt",
"allan","nelson","guy","clayton","hugh","max","dwayne","dwight","armando","felix",
"jimmie","everett","jordan","ian","wallace","ken","bob","jaime","casey","alfredo",
"alberto","dave","ivan","johnnie","sidney","byron","julian","isaac","morris","clifton",
"willard","daryl","ross","virgil","andy","marshall","salvador","perry","kirk",
"sergio","marion","tracy","seth","kent","terrance","rene","eduardo","terrence",
"enrique","freddie","wade", "mary","patricia","linda","barbara","elizabeth",
"jennifer","maria","susan","margaret","dorothy","lisa","nancy","karen","betty",
"helen","sandra","donna","carol","ruth","sharon","michelle","laura","sarah",
"kimberly","deborah","jessica","shirley","cynthia","angela","melissa","brenda",
"amy","anna","rebecca","virginia","kathleen","pamela","martha","debra","amanda",
"stephanie","carolyn","christine","marie","janet","catherine","frances","ann",
"joyce","diane","alice","julie","heather","teresa","doris","gloria","evelyn",
"jean","cheryl","mildred","katherine","joan","ashley","judith","rose","janice",
"kelly","nicole","judy","christina","kathy","theresa","beverly","denise","tammy",
"irene","jane","lori","rachel","marilyn","andrea","kathryn","louise","sara",
"anne","jacqueline","wanda","bonnie","julia","ruby","lois","tina","phyllis",
"norma","paula","diana","annie","lillian","emily","robin","peggy","crystal",
"gladys","rita","dawn","connie","florence","tracy","edna","tiffany","carmen",
"rosa","cindy","grace","wendy","victoria","edith","kim","sherry","sylvia",
"josephine","thelma","shannon","sheila","ethel","ellen","elaine","marjorie",
"carrie","charlotte","monica","esther","pauline","emma","juanita","anita",
"rhonda","hazel","amber","eva","debbie","april","leslie","clara","lucille",
"jamie","joanne","eleanor","valerie","danielle","megan","alicia","suzanne",
"michele","gail","bertha","darlene","veronica","jill","erin","geraldine",
"lauren","cathy","joann","lorraine","lynn","sally","regina","erica","beatrice",
"dolores","bernice","audrey","yvonne","annette","june","samantha","marion",
"dana","stacy","ana","renee","ida","vivian","roberta","holly","brittany",
"melanie","loretta","yolanda","jeanette","laurie","katie","kristen","vanessa",
"alma","sue","elsie","beth","jeanne","vicki","carla","tara","rosemary","eileen",
"terri","gertrude","lucy","tonya","ella","stacey","wilma","gina","kristin",
"jessie","natalie","agnes","vera","willie","charlene","bessie","delores",
"melinda","pearl","arlene","maureen","colleen","allison","tamara","joy",
"georgia","constance","lillie","claudia","jackie","marcia","tanya","nellie",
"minnie","marlene","heidi","glenda","lydia","viola","courtney","marian",
"stella","caroline","dora","jo","vickie","mattie","terry","maxine","irma",
"mabel","marsha","myrtle","lena","christy","deanna","patsy","hilda","gwendolyn",
"jennie","nora","margie","nina","cassandra","leah","penny","kay","priscilla",
"naomi","carole","brandy","olga","billie","dianne","tracey","leona","jenny",
"felicia","sonia","miriam","velma","becky","bobbie","violet","kristina","toni",
"misty","mae","shelly","daisy","ramona","sherri","erika","katrina","claire",
"lindsey","lindsay","geneva","guadalupe","belinda","margarita","sheryl","cora",
"faye","ada","natasha","sabrina","isabel","marguerite","hattie","harriet",
"molly","cecilia","kristi","brandi","blanche","sandy","rosie","joanna","iris",
"eunice","angie","inez","lynda","madeline","amelia","alberta","genevieve",
"monique","jodi","janie","maggie","kayla","sonya","jan","lee","kristine",
"candace","fannie","maryann","opal","alison","yvette","melody","luz","susie",
"olivia","flora","shelley","kristy","mamie","lula","lola","verna","beulah",
"antoinette","candice","juana","jeannette","pam","kelli","hannah","whitney",
"bridget","karla","celia","latoya","patty","shelia","gayle","della","vicky",
"lynne","sheri","marianne","kara","jacquelyn","erma","blanca","myra","leticia",
"pat","krista","roxanne","angelica","johnnie","robyn","francis","adrienne",
"rosalie","alexandra","brooke","bethany","sadie","bernadette","traci","jody",
"kendra","jasmine","nichole","rachael","chelsea","mable","ernestine","muriel",
"marcella","elena","krystal","angelina","nadine","kari","estelle","dianna",
"paulette","lora","mona","doreen","rosemarie","angel","desiree","antonia",
"hope","ginger","janis","betsy","christie","freda","mercedes","meredith",
"lynette","teri","cristina","eula","leigh","meghan","sophia","eloise",
"rochelle","gretchen","cecelia","raquel","henrietta","alyssa","jana","kelley",
"gwen","kerry","jenna","tricia","laverne","olive","alexis","tasha","silvia",
"elvira","casey","delia","sophie","kate","patti","lorena","kellie","sonja",
"lila","lana","darla","may","mindy","essie","mandy","lorene","elsa","josefina",
"jeannie","miranda","dixie","lucia","marta","faith","lela","johanna","shari",
"camille","tami","shawna","elisa","ebony","melba","ora","nettie","tabitha",
"ollie","jaime","winifred","kristie","marina","alisha","aimee","rena","myrna",
"marla","tammie","latasha","bonita","patrice","ronda","sherrie","addie",
"francine","deloris","stacie","adriana","cheri","shelby","abigail","celeste",
"jewel","cara","adele","rebekah","lucinda","dorthy","chris","effie","trina",
"reba","shawn","sallie","aurora","lenora","etta","lottie","kerri","trisha",
"nikki","estella","francisca","josie","tracie","marissa","karin","brittney",
"janelle","lourdes","laurel","helene","fern","elva","corinne","kelsey","ina",
"bettie","elisabeth","aida","caitlin","ingrid","iva","eugenia","christa",
"goldie","cassie","maude","jenifer","therese","frankie","dena","lorna","janette",
"latonya","candy","morgan","consuelo","tamika","rosetta","debora","cherie",
"polly","dina","jewell","fay","jillian","dorothea","nell","trudy","esperanza",
"patrica","kimberley","shanna","helena","carolina","cleo","stefanie","rosario",
"ola","janine","mollie","lupe","alisa","lou","maribel","susanne","bette",
"susana","elise","cecile","isabelle","lesley","jocelyn","paige","joni",
"rachelle","leola","daphne","alta","ester","petra","graciela","imogene",
"jolene","keisha","lacey","glenna","gabriela","keri","ursula","lizzie",
"kirsten","shana","adeline","mayra","jayne","jaclyn","gracie","sondra",
"carmela","marisa","rosalind","charity","tonia","beatriz","marisol","clarice",
"jeanine","sheena","angeline","frieda","lily","robbie","shauna","millie",
"claudette","cathleen","angelia","gabrielle","autumn","katharine","summer",
"jodie","staci","lea","christi","jimmie","justine","elma","luella","margret",
"dominique","socorro","rene","martina","margo","mavis","callie","bobbi",
"maritza","lucile","leanne","jeannine","deana","aileen","lorie","ladonna",
"willa","manuela","gale","selma","dolly","sybil","abby","lara","dale","ivy",
"dee","winnie","marcy","luisa","jeri","magdalena","ofelia","meagan","audra",
"matilda","leila","cornelia","bianca","simone","bettye","randi","virgie",
"latisha","barbra","georgina","eliza","leann","bridgette","rhoda","haley",
"adela","nola","bernadine","flossie","ila","greta","ruthie","nelda","minerva",
"lilly","terrie","letha","hilary","estela","valarie","brianna","rosalyn",
"earline","catalina","ava","mia","clarissa","lidia","corrine","alexandria",
"concepcion","tia","sharron","rae","dona","ericka","jami","elnora","chandra",
"lenore","neva","marylou","melisa","tabatha","serena","avis","allie","sofia",
"jeanie","odessa","nannie","harriett","loraine","penelope","milagros","emilia",
"benita","allyson","ashlee","tania","tommie","esmeralda","karina","eve",
"pearlie","zelma","malinda","noreen","tameka","saundra","hillary","amie",
"althea","rosalinda","jordan","lilia","alana","gay","clare","alejandra",
"elinor","michael","lorrie","jerri","darcy","earnestine","carmella","taylor",
"noemi","marcie","liza","annabelle","louisa","earlene","mallory","carlene",
"nita","selena","tanisha","katy","julianne","john","lakisha","edwina",
"maricela","margery","kenya","dollie","roxie","roslyn","kathrine","nanette",
"charmaine","lavonne","ilene","kris","tammi","suzette","corine","kaye","jerry",
"merle","chrystal","lina","deanne","lilian","juliana","aline","luann","kasey",
"maryanne","evangeline","colette","melva","lawanda","yesenia","nadia","madge",
"kathie","eddie","ophelia","valeria","nona","mitzi","mari","georgette",
"claudine","fran","alissa","roseann","lakeisha","susanna","reva","deidre",
"chasity","sheree","carly","james","elvia","alyce","deirdre","gena","briana",
"araceli","katelyn","rosanne","wendi","tessa","berta","marva","imelda",
"marietta","marci","leonor","arline","sasha","madelyn","janna","juliette",
"deena","aurelia","josefa","augusta","liliana","young","christian","lessie",
"amalia","savannah","anastasia","vilma","natalia","rosella","lynnette",
"corina","alfreda","leanna","carey","amparo","coleen","tamra","aisha","wilda",
"karyn","cherry","queen","maura","mai","evangelina","rosanna","hallie","erna",
"enid","mariana","lacy","juliet","jacklyn","freida","madeleine","mara","hester",
"cathryn","lelia","casandra","bridgett","angelita","jannie","dionne","annmarie",
"katina","beryl","phoebe","millicent","katheryn","diann","carissa","maryellen",
"liz","lauri","helga","gilda","adrian","rhea","marquita","hollie","tisha",
"tamera","angelique","francesca","britney","kaitlin","lolita","florine",
"rowena","reyna","twila","fanny","janell","ines","concetta","bertie","alba",
"brigitte","alyson","vonda","pansy","elba","noelle","letitia","kitty","deann",
"brandie","louella","leta","felecia","sharlene","lesa","beverley","robert",
"isabella","herminia","terra","celina"
]


class _StopWordsWrapper(object):
    """A mechanism for adding / managing custom stop words
        Parameters
        ----------
        cache_dir : str
           folder where the model will be saved
        stop_words : list
           a list of strings
    """
    _wrapper_type = "stop_words"

    def __init__(self, cache_dir='/tmp/'):
        self.cache_dir = cache_dir

    def save(self, name, stop_words):
        """Allow to save the stop_words list of strings with joblib.save
           under $CACHE_DIR/stop_words/<name>.pkl

            Parameters
            ----------
            name : str
                stop words name / identifier
            stop_words : list
                list of stop words
        """

        self.stop_words = stop_words # list of stop words

        self.model_dir = os.path.join(self.cache_dir, 'stop_words')
        self.name = os.path.join(self.model_dir, name + '.pkl') # the name (tag) for custom stop words list
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # dump(self.stop_words, self.name)
        dump(self.stop_words, self.name)

    def load(self, name):
        """Allow to retrive the stop_words list of strings
        """
        # self.name = name # the name of stop words list that must be loaded
        self.stop_words = load(self.name)
        return (self.stop_words)
