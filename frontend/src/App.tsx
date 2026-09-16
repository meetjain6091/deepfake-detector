import React, { useState } from "react";
import axios from "axios";

interface Probabilities {
  fake: number;
  real: number;
}

interface Results {
  label: "REAL" | "FAKE";
  confidence_percentage: number;
  probabilities: Probabilities;
}

interface PipelineMetadata {
  num_faces: number;
  bbox: number[];
  left_eye: number[];
  right_eye: number[];
  angle: number;
}

interface Visualizations {
  original: string;
  bounding_box: string;
  aligned_crop: string;
  resized_224: string;
}

interface IntermediateStages {
  metadata: PipelineMetadata;
  visualizations: Visualizations;
}

interface ApiResponse {
  status: string;
  log_id: number;
  filename: string;
  results: Results;
  intermediate_stages: IntermediateStages;
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [apiData, setApiData] = useState<ApiResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
      setApiData(null);
      setErrorMessage(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setErrorMessage("Please select an image file first.");
      return;
    }

    setLoading(true);
    setErrorMessage(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post<ApiResponse>(
        "http://127.0.0.1:8000/predict",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        },
      );
      setApiData(response.data);
    } catch (err: any) {
      console.error("Inference Error:", err);
      const detail =
        err.response?.data?.detail || "Inference execution failed.";
      setErrorMessage(detail);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        backgroundColor: "#0d1117",
        color: "#c9d1d9",
        minHeight: "100vh",
        padding: "40px 20px",
        fontFamily:
          "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
      }}
    >
      <div style={{ maxWidth: "1000px", margin: "0 auto" }}>
        <header style={{ textAlign: "center", marginBottom: "30px" }}>
          <h1
            style={{
              color: "#58a6ff",
              fontSize: "2.2rem",
              marginBottom: "8px",
            }}
          >
            Dual-Paradigm Deepfake Detector
          </h1>
          <p style={{ color: "#8b949e" }}>
            Real-time visual feature extraction, eye-alignment, and
            quantum-inspired classification pipeline
          </p>
        </header>

        {/* Input Control Section */}
        <div
          style={{
            backgroundColor: "#161b22",
            border: "1px solid #30363d",
            borderRadius: "10px",
            padding: "25px",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: "20px",
          }}
        >
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            style={{
              color: "#c9d1d9",
              backgroundColor: "#21262d",
              padding: "10px",
              borderRadius: "6px",
              border: "1px solid #30363d",
              cursor: "pointer",
            }}
          />

          <button
            onClick={handleUpload}
            disabled={loading || !file}
            style={{
              padding: "12px 28px",
              backgroundColor: loading || !file ? "#21262d" : "#238636",
              color: loading || !file ? "#8b949e" : "#ffffff",
              border: "none",
              borderRadius: "6px",
              fontSize: "1rem",
              fontWeight: "bold",
              cursor: loading || !file ? "not-allowed" : "pointer",
              transition: "0.2s ease",
            }}
          >
            {loading ? "Analyzing Pipeline..." : "Run Detection"}
          </button>
        </div>

        {/* Error Notification */}
        {errorMessage && (
          <div
            style={{
              backgroundColor: "#3c1e1e",
              border: "1px solid #f85149",
              color: "#ff7b72",
              padding: "15px",
              borderRadius: "8px",
              marginTop: "20px",
              textAlign: "center",
            }}
          >
            {errorMessage}
          </div>
        )}

        {/* Inference Results Dashboard */}
        {apiData && (
          <div style={{ marginTop: "30px" }}>
            {/* Classification Summary Card */}
            <div
              style={{
                backgroundColor: "#161b22",
                border: "1px solid #30363d",
                borderRadius: "10px",
                padding: "20px",
                marginBottom: "25px",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                flexWrap: "wrap",
                gap: "15px",
              }}
            >
              <div>
                <span style={{ fontSize: "1rem", color: "#8b949e" }}>
                  Prediction Result:
                </span>
                <h2
                  style={{
                    fontSize: "2.5rem",
                    margin: "5px 0 0 0",
                    color:
                      apiData.results.label === "FAKE" ? "#f85149" : "#3fb950",
                  }}
                >
                  {apiData.results.label}
                </h2>
              </div>
              <div>
                <p style={{ margin: "4px 0", fontSize: "1.1rem" }}>
                  <strong>Confidence:</strong>{" "}
                  {apiData.results.confidence_percentage}%
                </p>
                <p style={{ margin: "4px 0", color: "#8b949e" }}>
                  Fake Probability:{" "}
                  {(apiData.results.probabilities.fake * 100).toFixed(2)}% |
                  Real Probability:{" "}
                  {(apiData.results.probabilities.real * 100).toFixed(2)}%
                </p>
              </div>
            </div>

            {/* Intermediate Processing Pipeline Visualization */}
            <h3 style={{ color: "#58a6ff", marginBottom: "15px" }}>
              Intermediate Pipeline Stages
            </h3>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))",
                gap: "15px",
              }}
            >
              {/* Stage 1 */}
              <div
                style={{
                  backgroundColor: "#161b22",
                  border: "1px solid #30363d",
                  borderRadius: "8px",
                  padding: "12px",
                  textAlign: "center",
                }}
              >
                <p
                  style={{
                    fontSize: "0.85rem",
                    fontWeight: "bold",
                    color: "#8b949e",
                    marginBottom: "8px",
                  }}
                >
                  1. ORIGINAL INPUT
                </p>
                <img
                  src={apiData.intermediate_stages.visualizations.original}
                  alt="Original"
                  style={{
                    width: "100%",
                    height: "180px",
                    objectFit: "contain",
                    borderRadius: "4px",
                  }}
                />
              </div>

              {/* Stage 2 */}
              <div
                style={{
                  backgroundColor: "#161b22",
                  border: "1px solid #30363d",
                  borderRadius: "8px",
                  padding: "12px",
                  textAlign: "center",
                }}
              >
                <p
                  style={{
                    fontSize: "0.85rem",
                    fontWeight: "bold",
                    color: "#8b949e",
                    marginBottom: "8px",
                  }}
                >
                  2. MTCNN BBOX & EYES
                </p>
                <img
                  src={apiData.intermediate_stages.visualizations.bounding_box}
                  alt="BBox Detection"
                  style={{
                    width: "100%",
                    height: "180px",
                    objectFit: "contain",
                    borderRadius: "4px",
                  }}
                />
              </div>

              {/* Stage 3 */}
              <div
                style={{
                  backgroundColor: "#161b22",
                  border: "1px solid #30363d",
                  borderRadius: "8px",
                  padding: "12px",
                  textAlign: "center",
                }}
              >
                <p
                  style={{
                    fontSize: "0.85rem",
                    fontWeight: "bold",
                    color: "#8b949e",
                    marginBottom: "8px",
                  }}
                >
                  3. ALIGNED FACE (ROTATED{" "}
                  {apiData.intermediate_stages.metadata.angle}°)
                </p>
                <img
                  src={apiData.intermediate_stages.visualizations.aligned_crop}
                  alt="Aligned Face"
                  style={{
                    width: "100%",
                    height: "180px",
                    objectFit: "contain",
                    borderRadius: "4px",
                  }}
                />
              </div>

              {/* Stage 4 */}
              <div
                style={{
                  backgroundColor: "#161b22",
                  border: "1px solid #30363d",
                  borderRadius: "8px",
                  padding: "12px",
                  textAlign: "center",
                }}
              >
                <p
                  style={{
                    fontSize: "0.85rem",
                    fontWeight: "bold",
                    color: "#8b949e",
                    marginBottom: "8px",
                  }}
                >
                  4. MODEL TENSOR (224x224)
                </p>
                <img
                  src={apiData.intermediate_stages.visualizations.resized_224}
                  alt="Model Input"
                  style={{
                    width: "100%",
                    height: "180px",
                    objectFit: "contain",
                    borderRadius: "4px",
                  }}
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
