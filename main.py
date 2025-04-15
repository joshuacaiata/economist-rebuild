#!/usr/bin/env python3
import os
import argparse
import sys
import yaml
from training.two_phase_trainer import TwoPhaseTrainer
from training.three_phase_trainer import ThreePhaseTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml",
                      help="Path to the configuration file")
    parser.add_argument("--use-gpu", action="store_true", default=False,
                      help="Whether to use GPU for training")
    parser.add_argument("--phase", type=int, default=0,
                      help="Which phase to run (0=all, 1=phase1, 2=phase2, 3=phase3)")
    parser.add_argument("--training-type", type=str, choices=["two_phase", "three_phase"],
                      default="three_phase", help="Type of training to run")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found!")
        sys.exit(1)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['use_gpu'] = args.use_gpu
    
    # Create appropriate trainer based on training type
    if args.training_type == "two_phase":
        trainer = TwoPhaseTrainer(config)
        
        # Run training based on specified phase
        if args.phase == 1:
            trainer.phase_one()
        elif args.phase == 2:
            trainer.phase_two()
        else:
            trainer.train_two_phase()
            
    elif args.training_type == "three_phase":
        print(f"="* 50)
        print("Training three phase system")
        print(f"="* 50)
        trainer = ThreePhaseTrainer(config)
        
        # Run training based on specified phase
        if args.phase == 1:
            trainer.phase_one_and_two()  # Phases 1&2 are combined in three-phase trainer
        elif args.phase == 3:
            trainer.phase_three()
        else:
            trainer.train_three_phase()
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
