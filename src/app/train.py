#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .args import build_parser
from .loop import run_training

def main():
    parser = build_parser()
    args = parser.parse_args()
    run_training(args)

if __name__ == "__main__":
    main()
