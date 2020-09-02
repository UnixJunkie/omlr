(* Copyright (C) 2020, Francois Berenger

   Yamanishi laboratory,
   Department of Bioscience and Bioinformatics,
   Faculty of Computer Science and Systems Engineering,
   Kyushu Institute of Technology,
   680-4 Kawazu, Iizuka, Fukuoka, 820-8502, Japan.

   Train/test a MLR regressor *)

open Printf

module A = Array
module CLI = Minicli.CLI
module Gnuplot = Omlr.Gnuplot
module L = BatList
module Log = Dolog.Log
module MLR = Omlr.MLR
module Utls = Omlr.Utls

let train debug sep maybe_model_fn train_lines =
  let train_data = MLR.matrix_of_csv_lines ~sep train_lines in
  let model = MLR.train_model ~debug train_data in
  (match maybe_model_fn with
   | None -> MLR.dump_model_to_file "/dev/stdout" model;
   | Some fn -> MLR.dump_model_to_file fn model
  );
  model

let test sep model nb_test test_lines =
  let actual, test_data =
    let a = MLR.matrix_of_csv_lines ~sep test_lines in
    (* first col. is target value *)
    (A.to_list a.(0), Utls.transpose_matrix a) in
  assert(A.length test_data = nb_test);
  let preds =
    let predicted' =
      A.init nb_test (fun i ->
          MLR.predict_one model test_data.(i)
        ) in
    A.to_list predicted' in
  (actual, preds)

let train_test debug sep randomize skip_header train_portion maybe_model_fn input_fn =
  let train_lines, test_lines =
    let all_lines = MLR.read_csv_file ~randomize ~skip_header input_fn in
    Cpm.Utls.train_test_split train_portion all_lines in
  let nb_train, nb_test = L.(length train_lines, length test_lines) in
  Log.info "train: %d test: %d total: %d"
    nb_train nb_test (nb_train + nb_test);
  (* train *)
  let model = train debug sep maybe_model_fn train_lines in
  (* test *)
  test sep model nb_test test_lines

let train_test_nfolds debug sep randomize skip_header nfolds input_fn =
  let all_lines = MLR.read_csv_file ~randomize ~skip_header input_fn in
  let train_tests = Cpm.Utls.cv_folds nfolds all_lines in
  let actual_preds =
    L.map (fun (train_lines, test_lines) ->
        let model = train debug sep None train_lines in
        test sep model (L.length test_lines) test_lines
      ) train_tests in
  let actual, preds = L.split actual_preds in
  (L.concat actual, L.concat preds)

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
               TODO [-l|--load <filename>]: restore model from file\n  \
               TODO [-o <filename>]: predictions output file\n  \
               [--no-shuffle]: do not randomize input lines\n  \
               [--no-header]: CSV file has no header\n  \
               [--no-plot]: don't call gnuplot\n  \
               [-d <char>]: field delimited in CSV file (default=',')\n  \
               [-v]: verbose/debug mode\n  \
               [-h|--help]: show this message\n"
        Sys.argv.(0);
      exit 1
    end;
  (* CLI options *)
  let randomize = not (CLI.get_set_bool ["--no-shuffle"] args) in
  let skip_header = not (CLI.get_set_bool ["--no-header"] args) in
  let no_plot = CLI.get_set_bool ["--no-plot"] args in
  let train_portion = CLI.get_float_def ["-p"] args 0.8 in
  let maybe_model_fn = CLI.get_string_opt ["-s"] args in
  let debug = CLI.get_set_bool ["-v"] args in
  let sep = CLI.get_char_def ["-d"] args ',' in
  let nfolds = CLI.get_int_def ["-n";"--NxCV"] args 1 in
  let input_fn = CLI.get_string ["-i"] args in
  CLI.finalize (); (* ------------------------------------------------------ *)
  let actual, preds =
    if nfolds <= 1 then
      train_test debug sep randomize skip_header train_portion maybe_model_fn input_fn
    else (* nfolds > 1 *)
      train_test_nfolds debug sep randomize skip_header nfolds input_fn
  in
  let r2 = Cpm.RegrStats.r2 actual preds in
  let r2_str = sprintf "R2: %.3f" r2 in
  (if not no_plot then Gnuplot.regr_plot r2_str actual preds);
  Log.info "%s" r2_str;
  ()

let () = main ()
