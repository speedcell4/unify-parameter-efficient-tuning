exp1:
	STUDY=res lr=1e-4 decompose_embed=True use_res=True fc_size=4 num_rotations=2 bash exps/run_house.sh

exp2:
	STUDY=res lr=3e-4 decompose_embed=True use_res=True fc_size=4 num_rotations=2 bash exps/run_house.sh

exp3:
	STUDY=res lr=5e-4 decompose_embed=True use_res=True fc_size=4 num_rotations=2 bash exps/run_house.sh

exp4:
	STUDY=res lr=7e-4 decompose_embed=True use_res=True fc_size=4 num_rotations=2 bash exps/run_house.sh

exp5:
	STUDY=res lr=9e-4 decompose_embed=True use_res=True fc_size=4 num_rotations=2 bash exps/run_house.sh

exp6:
	STUDY=res lr=1e-3 decompose_embed=True use_res=True fc_size=4 num_rotations=2 bash exps/run_house.sh

exp7:
	STUDY=res lr=3e-3 decompose_embed=True use_res=True fc_size=4 num_rotations=2 bash exps/run_house.sh

exp8:
	STUDY=res lr=5e-3 decompose_embed=True use_res=True fc_size=4 num_rotations=2 bash exps/run_house.sh

all: exp1 exp2 exp3 exp4 exp5 exp6 exp7 exp8
