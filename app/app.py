import os
import streamlit as st
import numpy as np
import pickle
# --- LOAD MODEL --- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

with open(model_path, "rb") as f:
    w, b, mean, std, y_mean, y_std = pickle.load(f)
