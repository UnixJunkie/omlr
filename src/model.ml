(* Copyright (C) 2020, Francois Berenger

   Yamanishi laboratory,
   Department of Bioscience and Bioinformatics,
   Faculty of Computer Science and Systems Engineering,
   Kyushu Institute of Technology,
   680-4 Kawazu, Iizuka, Fukuoka, 820-8502, Japan.

   Train/test a MLR regressor *)

open Printf

module CLI = Minicli.CLI
module L = BatList
module Log = Dolog.Log
module MLR = Omlr.MLR
module Utls = Omlr.Utls

let main () =
  Log.(set_log_level DEBUG);
  Log.color_on ();
  Log.info "start";
  let argc, args = CLI.init () in
  let show_help = CLI.get_set_bool ["-h";"--help"] args in
  if argc = 1 || show_help then
    begin
      eprintf "usage:\n\
               %s\n  \
               [-i <input.csv>]: input CSV file\n  \
               [--NxCV <int>]: number of folds of cross validation\n  \
               [-s|--save <filename>]: save model to file\n  \
               [-l|--load <filename>]: restore model from file\n  \
               [-o <filename>]: predictions output file\n  \
               [--no-plot]: don't call gnuplot\n  \
               [-v]: verbose/debug mode\n  \
               [-h|--help]: show this message\n"
        Sys.argv.(0);
      exit 1
    end;
  let input_fn = CLI.get_string ["-i"] args in
  let data = MLR.load_csv_file true ',' input_fn in
  let _norm_params, normalized = MLR.normalize_features data in
  let _ = MLR.train_model_gd normalized in
  ()

let () = main ()
