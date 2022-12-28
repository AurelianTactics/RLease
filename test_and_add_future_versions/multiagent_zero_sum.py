from trueskill import Rating, quality_1vs1, rate_1vs1
from rlease_utils import print_rankings_wld_dict, print_head_to_head_dict, get_save_path_and_make_save_directory

"""
Instantiate Class and call run to analyze results for multiple agents
Unclear at this time whether for different types of agents and envs to send args here or inherit from here
"""

class MultiAgentZeroSum():
    def __init__(self, args_dict):
        # unpack args

        # store various dicts in results_dict:
        # agents_dict: agents to evaluate
        # rankings_dict: TrueSkill/ELO/etc like rankings
        # head_to_head_dict: individual game results, stored in a way to make HTH match up analysis easier
        # wld_dict: store overall game results for an agent 
        # stats_dict:
        self.AGENTS_DICT_KEY = 'AGENTS_DICT_KEY'
        self.RANKINGS_DICT_KEY = 'RANKINGS_DICT_KEY'
        self.HEAD_TO_HEAD_DICT_KEY = 'HEAD_TO_HEAD_DICT_KEY'
        self.WLD_DICT_KEY = 'WLD_DICT_KEY'
        self.STATS_DICT_KEY = 'STATS_DICT_KEY'
        self.GAME_NUMBER_KEY = 'GAME_NUMBER_KEY'
        self.MAZS_dict = self.load_or_create_mazs_dict(args_dict)

        # print results of prior MAZS DICT and quit
        self.print_and_quit = args_dict['print_and_quit']
        self.print_mazs_dict(self.print_and_quit, args_dict)
        

    def run(self, args_dict):
        # MAZS_Dict loaded, created and amended in init
        if not self.print_and_quit:
            self.MAZS_Dict = self.play_match(args_dict)
            self.save_results()
            self.print_mazs_dict(True, args_dict)
            

    def play_match(self, args_dict):
        # iterate through opponent combinations
        agent_names = self.MAZS_Dict[self.AGENTS_DICT_KEY].keys()
        game_number = self.get_max_game_number()
        for i in range(len(agent_names)-1):
            current_agent_name = agent_names[i]
            for j in range(i+1, len(agent_names)):
                opp_agent_name = agent_names[j]
                # check if these bots have already played
                if check_if_already_played(match_up_dict, current_agent_name, opp_agent_name):
                    continue
                # play args many games
                for _ in range(args_dict.num_games):
                    # game_number used as key for some dicts
                    game_number += 1
                    # special args and stats for different envs 
                    temp_stats_dict, temp_hth_dict = get_game_result(game_number) # env specific
                    # update dicts with results
                    #rankings, wl, head to head, if already played
                    update_dicts_with_results(temp_stats_dict, temp_hth_dict, game_number)

    def get_max_game_number(self):
        return self.MAZS_Dict[self.GAME_NUMBER_KEY]

    def save_results(self, args_dict):
        # placeholder
        # save the file
        file_dir = "~/RLEase_results/eval_multiagent_zero_sum_results/"
        file_name = "eval_mazs_{}_save_{}.pickle".format(args_dict['project_name'], int(time.time()))
        save_path = get_save_path_and_make_save_directory(file_name, file_dir)
        with open(save_path, 'wb') as handle:
            pickle.dump(self.MAZS_Dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def print_mazs_dict(is_print, args_dict):
        if is_print:
            # print mazs_dict
            # placeholders
            print_rankings_wld_dict(self.MAZS_Dict[self.WLD_DICT_KEY], self.MAZS_Dict[self.AGENTS_DICT_KEY], 
                self.MAZS_Dict[self.AGENTS_DICT_KEY])
            print_head_to_head_dict(self.MAZS_Dict[self.HEAD_TO_HEAD_DICT_KEY], True)


    def load_or_create_mazs_dict(self, args_dict):
        # create or load new agents
        # can either combine with existing agents dict if loading prior results
        # or use to create a new mazs dict
        # load agents: either inputted directly in yaml or a yaml file to load or a directory 
        # to scan for agent files
        if 'new_agents_dict' in args_dict:
            pass
        elif 'new_agents_path' in args_dict:
            pass
        elif 'new_agent_file' in args_dict:
            new_agents_dict = yaml.safe_load(args_dict['new_agent_file'])
        else:
            print("ERROR: No agents specified to load, quitting")
            quit()

        # load prior results dict if key
        if 'results_filepath' in args_dict:
            MAZS_dict = yaml.safe_load(args_dict['results_filepath'])
            # add new agents to existing agents
            MAZS_dict = self.add_new_agents_to_existing_dict(MAZS_dict, new_agents_dict)
        else:
            MAZS_dict = self.create_new_mazs_dict(args_dict, new_agents_dict)
        
        return MAZS_dict

    def add_new_agents_to_existing_dict(self, MAZS_dict, new_agents_dict):
        agent_names = list(new_agents_dict.keys())
        # game numbers are the keys
        # incremented before an entry is added to the dict 

        for an in agent_names:
            if an in MAZS_dict[self.AGENTS_DICT_KEY] or an in MAZS_dict[self.STATS_DICT_KEY] or \
            an in MAZS_dict[self.RESULTS_DICT_KEY] or an in MAZS_dict[self.WLD_DICT_KEY] or \
            an in MAZS_dict[self.HEAD_TO_HEAD_DICT_KEY]:
                print("WARNING: agent already exists {}, not adding it to dict ".format(an))
                # to do: maybe add option to make the agent again with a timestamp?
                continue
                # an = an + "_{}".format(int(time.time()))
                # print("adding timestamp to agent name {}".format(an))

            MAZS_dict[self.AGENTS_DICT_KEY][an] = new_agents_dict[an]
            MAZS_dict[self.RESULTS_DICT_KEY][an] = get_eval_object(args_dict)
            MAZS_dict[self.WLD_DICT_KEY][an] = {
                    GAME_RESULT_WIN: 0,
                    GAME_RESULT_LOSS: 0,
                    GAME_RESULT_DRAW: 0,
                    GAME_SCORE_KEY: [],
                    GAME_SCORE_OPP_KEY: [],
                    GAME_HOME_KEY: [],
                }

        return MAZS_dict


    def create_new_mazs_dict(self, args_dict, agent_dict):
        rankings_dict = {}
        wld_dict = {}
        for k, v in agent_dict.items():
            rankings_dict[k] = get_eval_object(args_dict)
            wld_dict[k] = {
                GAME_RESULT_WIN: 0,
                GAME_RESULT_LOSS: 0,
                GAME_RESULT_DRAW: 0,
                GAME_SCORE_KEY: [],
                GAME_SCORE_OPP_KEY: [],
                GAME_HOME_KEY: [],
            }

        MAZS_dict[self.AGENTS_DICT_KEY] = agent_dict
        MAZS_dict[self.RANKINGS_DICT_KEY] = rankings_dict
        MAZS_dict[self.WLD_DICT_KEY] = wld_dict
        MAZS_dict[self.STATS_DICT_KEY] = {}
        MAZS_dict[self.HEAD_TO_HEAD_DICT_KEY] = {} 

        MAZS_Dict[self.GAME_NUMBER_KEY] = 0

        return MAZS_dict    

    def get_eval_type(self, args_dict):
        '''
        For new agents, initialize the rating object based on the argument
        This rating object will change based on the game results being fed into it
        '''
        if args.eval_type == EVAL_TYPE_TRUESKILL:
            return Rating() # new trueskill rating
        else:
            return Rating() # new trueskill rating object
