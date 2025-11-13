import axios from "axios";

export const API_URL = "http://127.0.0.1:8000".replace(/\/$/, "");
axios.defaults.withCredentials = true;
const api = axios.create({
  baseURL: API_URL,
});

export default api;
