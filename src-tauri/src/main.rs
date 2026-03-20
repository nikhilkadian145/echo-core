// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod audio;
mod ws;
mod process;
mod commands;

fn main() {
    echo_core_tmp_lib::run()
}
