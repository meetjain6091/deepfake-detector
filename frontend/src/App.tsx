import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./App.css";

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

interface ModalState {
  title: string;
  stageName: string;
  imageSrc: string;
  accentColor: string;
  telemetry: { label: string; value: string }[];
  description: string;
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [apiData, setApiData] = useState<ApiResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [activeModal, setActiveModal] = useState<ModalState | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Close Modal on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setActiveModal(null);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const handleFileSelected = (selectedFile: File) => {
    if (!selectedFile.type.startsWith("image/")) {
      setErrorMessage("Please upload a valid image file (JPEG, PNG, etc.)");
      return;
    }
    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setApiData(null);
    setErrorMessage(null);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
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
        err.response?.data?.detail ||
        "Inference execution failed. Check backend server.";
      setErrorMessage(detail);
    } finally {
      setLoading(false);
    }
  };

  const resetState = () => {
    setFile(null);
    setPreviewUrl(null);
    setApiData(null);
    setErrorMessage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div
      style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}
    >
      {/* Top Header Bar */}
      <header
        style={{
          borderBottom: "1px solid rgba(255, 255, 255, 0.1)",
          backgroundColor: "rgba(3, 7, 18, 0.85)",
          backdropFilter: "blur(16px)",
          position: "sticky",
          top: 0,
          zIndex: 50,
          padding: "16px 36px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div
            style={{
              width: "12px",
              height: "12px",
              borderRadius: "50%",
              backgroundColor: "#2373F4",
              boxShadow: "0 0 14px #2373F4",
            }}
          />
          <h2
            style={{
              fontSize: "1.25rem",
              fontWeight: "800",
              letterSpacing: "-0.5px",
              margin: 0,
              color: "#f8fafc",
            }}
          >
            DUAL-PARADIGM{" "}
            <span style={{ color: "#65D0F4" }}>DEEPFAKE DETECTOR</span>
          </h2>
        </div>
        <div
          style={{
            fontSize: "0.8rem",
            padding: "6px 14px",
            borderRadius: "20px",
            background: "rgba(101, 208, 244, 0.12)",
            border: "1px solid rgba(101, 208, 244, 0.35)",
            color: "#65D0F4",
            fontWeight: 600,
          }}
        >
          MTCNN Alignment & Vision Pipeline v2.4
        </div>
      </header>

      {/* Main Container */}
      <main
        style={{
          flex: 1,
          maxWidth: "1240px",
          width: "100%",
          margin: "0 auto",
          padding: "40px 20px",
        }}
      >
        {/* Hero Title */}
        <div style={{ textAlign: "center", marginBottom: "40px" }}>
          <h1
            style={{
              fontSize: "2.8rem",
              fontWeight: 800,
              margin: "0 0 12px 0",
              letterSpacing: "-1px",
              color: "#f8fafc",
            }}
          >
            Facial Forensics & Neural Pipeline
          </h1>
          <p
            style={{
              color: "#94a3b8",
              fontSize: "1.1rem",
              maxWidth: "700px",
              margin: "0 auto",
            }}
          >
            Automated multi-stage facial landmark extraction, affine eye
            alignment, spatial cropping, and dual-paradigm deep classification.
          </p>
        </div>

        {/* Upload Workspace */}
        <div
          className="glass-panel animate-pulse-glow"
          style={{
            borderRadius: "20px",
            padding: "36px",
            marginBottom: "36px",
          }}
        >
          <input
            type="file"
            accept="image/*"
            ref={fileInputRef}
            onChange={handleFileChange}
            style={{ display: "none" }}
          />

          {!previewUrl ? (
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              style={{
                border: `2px dashed ${isDragging ? "#2373F4" : "#334155"}`,
                borderRadius: "16px",
                padding: "50px 20px",
                textAlign: "center",
                cursor: "pointer",
                backgroundColor: isDragging
                  ? "rgba(35, 115, 244, 0.08)"
                  : "rgba(15, 23, 42, 0.4)",
                transition: "all 0.25s ease",
              }}
            >
              <div style={{ fontSize: "3.2rem", marginBottom: "16px" }}>🔬</div>
              <h3
                style={{
                  fontSize: "1.25rem",
                  fontWeight: 600,
                  marginBottom: "8px",
                  color: "#f8fafc",
                }}
              >
                Drag & drop facial image, or{" "}
                <span style={{ color: "#65D0F4" }}>browse local storage</span>
              </h3>
              <p style={{ color: "#94a3b8", fontSize: "0.95rem" }}>
                Supports high-resolution PNG, JPG, or JPEG portrait inputs
              </p>
            </div>
          ) : (
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: "24px",
              }}
            >
              <div
                style={{
                  position: "relative",
                  borderRadius: "14px",
                  overflow: "hidden",
                  border: "1px solid #334155",
                  maxHeight: "320px",
                  backgroundColor: "#020617",
                }}
              >
                <img
                  src={previewUrl}
                  alt="Upload Preview"
                  style={{
                    maxHeight: "300px",
                    maxWidth: "100%",
                    objectFit: "contain",
                    display: "block",
                  }}
                />
                {loading && <div className="scanner-line" />}
              </div>

              <div
                style={{
                  display: "flex",
                  gap: "14px",
                  flexWrap: "wrap",
                  justifyContent: "center",
                }}
              >
                <button
                  onClick={handleUpload}
                  disabled={loading}
                  style={{
                    padding: "14px 36px",
                    backgroundColor: loading ? "#1e293b" : "#2373F4",
                    color: loading ? "#94a3b8" : "#ffffff",
                    border: "none",
                    borderRadius: "10px",
                    fontSize: "1rem",
                    fontWeight: 700,
                    cursor: loading ? "not-allowed" : "pointer",
                    boxShadow: loading
                      ? "none"
                      : "0 6px 20px rgba(35, 115, 244, 0.45)",
                    transition: "all 0.2s ease",
                  }}
                >
                  {loading ? (
                    <span
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "10px",
                      }}
                    >
                      <span
                        style={{
                          width: "16px",
                          height: "16px",
                          border: "2px solid #fff",
                          borderTopColor: "transparent",
                          borderRadius: "50%",
                          animation: "spin 0.8s linear infinite",
                        }}
                      />
                      Processing Pipeline Analytics...
                    </span>
                  ) : (
                    "⚡ Execute Deepfake Analysis"
                  )}
                </button>

                <button
                  onClick={resetState}
                  disabled={loading}
                  style={{
                    padding: "14px 22px",
                    backgroundColor: "transparent",
                    color: "#94a3b8",
                    border: "1px solid #334155",
                    borderRadius: "10px",
                    fontSize: "1rem",
                    cursor: "pointer",
                    transition: "all 0.2s ease",
                  }}
                >
                  Clear Selection
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Error Notification */}
        {errorMessage && (
          <div
            style={{
              backgroundColor: "rgba(185, 54, 54, 0.15)",
              border: "1px solid rgba(185, 54, 54, 0.5)",
              color: "#fca5a5",
              padding: "16px 22px",
              borderRadius: "12px",
              marginBottom: "30px",
              textAlign: "center",
              fontWeight: 600,
            }}
          >
            ⚠️ {errorMessage}
          </div>
        )}

        {/* Results Dashboard */}
        {apiData && (
          <div style={{ animation: "fadeIn 0.5s ease" }}>
            {/* Primary Classification Verdict Banner */}
            <div
              className="glass-panel"
              style={{
                borderRadius: "18px",
                padding: "32px",
                marginBottom: "40px",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                flexWrap: "wrap",
                gap: "24px",
                borderLeft: `6px solid ${
                  apiData.results.label === "FAKE" ? "#B93636" : "#7D9C65"
                }`,
              }}
            >
              <div>
                <span
                  style={{
                    fontSize: "0.85rem",
                    textTransform: "uppercase",
                    letterSpacing: "1.2px",
                    color: "#94a3b8",
                    fontWeight: 700,
                  }}
                >
                  Classification Verdict
                </span>
                <div
                  style={{
                    display: "flex",
                    alignItems: "baseline",
                    gap: "18px",
                    marginTop: "6px",
                  }}
                >
                  <h2
                    style={{
                      fontSize: "3.2rem",
                      fontWeight: 900,
                      margin: 0,
                      color:
                        apiData.results.label === "FAKE"
                          ? "#B93636"
                          : "#7D9C65",
                      letterSpacing: "-1px",
                    }}
                  >
                    {apiData.results.label}
                  </h2>
                  <span style={{ fontSize: "1.15rem", color: "#94a3b8" }}>
                    Confidence:{" "}
                    <strong style={{ color: "#f8fafc" }}>
                      {apiData.results.confidence_percentage}%
                    </strong>
                  </span>
                </div>
              </div>

              {/* Probability Breakdown Bar */}
              <div style={{ minWidth: "300px", flex: 1, maxWidth: "460px" }}>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    fontSize: "0.95rem",
                    marginBottom: "8px",
                    fontWeight: 600,
                  }}
                >
                  <span style={{ color: "#7D9C65" }}>
                    Real Probability:{" "}
                    {(apiData.results.probabilities.real * 100).toFixed(2)}%
                  </span>
                  <span style={{ color: "#B93636" }}>
                    Fake Probability:{" "}
                    {(apiData.results.probabilities.fake * 100).toFixed(2)}%
                  </span>
                </div>
                <div
                  style={{
                    height: "14px",
                    width: "100%",
                    backgroundColor: "#020617",
                    borderRadius: "7px",
                    overflow: "hidden",
                    display: "flex",
                    border: "1px solid rgba(255, 255, 255, 0.1)",
                  }}
                >
                  <div
                    style={{
                      width: `${apiData.results.probabilities.real * 100}%`,
                      backgroundColor: "#7D9C65",
                      transition: "width 0.8s cubic-bezier(0.4, 0, 0.2, 1)",
                    }}
                  />
                  <div
                    style={{
                      width: `${apiData.results.probabilities.fake * 100}%`,
                      backgroundColor: "#B93636",
                      transition: "width 0.8s cubic-bezier(0.4, 0, 0.2, 1)",
                    }}
                  />
                </div>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    fontSize: "0.8rem",
                    color: "#94a3b8",
                    marginTop: "8px",
                  }}
                >
                  <span>Log Hash ID: #{apiData.log_id}</span>
                  <span>
                    Faces Detected:{" "}
                    {apiData.intermediate_stages.metadata.num_faces}
                  </span>
                </div>
              </div>
            </div>

            {/* Intermediate Processing Pipeline: STRICT 2x2 MATRIX */}
            <div style={{ marginBottom: "30px" }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "flex-end",
                  marginBottom: "20px",
                }}
              >
                <div>
                  <h3
                    style={{
                      fontSize: "1.6rem",
                      fontWeight: 800,
                      margin: "0 0 6px 0",
                      color: "#f8fafc",
                    }}
                  >
                    Neural Feature & Preprocessing Matrix (2x2 Pipeline)
                  </h3>
                  <p
                    style={{ color: "#94a3b8", fontSize: "0.95rem", margin: 0 }}
                  >
                    Click on any stage card below to inspect full-screen
                    high-resolution diagnostics and telemetry.
                  </p>
                </div>
                <span
                  style={{
                    fontSize: "0.8rem",
                    color: "#65D0F4",
                    background: "rgba(101, 208, 244, 0.1)",
                    padding: "6px 12px",
                    borderRadius: "8px",
                    border: "1px solid rgba(101, 208, 244, 0.3)",
                  }}
                >
                  💡 Click image to inspect
                </span>
              </div>

              {/* STRICT 2x2 MATRIX GRID */}
              <div className="matrix-grid">
                {/* ROW 1 - STAGE 01 (Electric Blue Accent) */}
                <div
                  className="stage-card stage-1"
                  onClick={() =>
                    setActiveModal({
                      title: "Stage 01: Raw Input Image Parsing",
                      stageName: "STAGE 01",
                      imageSrc:
                        apiData.intermediate_stages.visualizations.original,
                      accentColor: "#2373F4",
                      description:
                        "Decodes incoming image bytes into standard uncompressed RGB channels, removing alpha transparency masks and scaling color matrix arrays.",
                      telemetry: [
                        { label: "Channel Standard", value: "3-Channel RGB" },
                        {
                          label: "File Format",
                          value:
                            apiData.filename.split(".").pop()?.toUpperCase() ||
                            "IMAGE",
                        },
                        { label: "Input Buffer", value: "Standardized" },
                        { label: "Memory Status", value: "Allocated" },
                      ],
                    })
                  }
                >
                  <div className="hover-overlay">
                    <span
                      style={{
                        color: "#fff",
                        fontWeight: 700,
                        fontSize: "0.95rem",
                        background: "#2373F4",
                        padding: "8px 16px",
                        borderRadius: "8px",
                      }}
                    >
                      🔍 Inspect High-Res
                    </span>
                  </div>

                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      marginBottom: "12px",
                    }}
                  >
                    <span
                      style={{
                        fontSize: "0.85rem",
                        fontWeight: 800,
                        color: "#2373F4",
                        letterSpacing: "1px",
                      }}
                    >
                      STAGE 01
                    </span>
                    <span
                      style={{
                        fontSize: "0.75rem",
                        color: "#2373F4",
                        fontWeight: 700,
                        background: "rgba(35, 115, 244, 0.18)",
                        padding: "3px 10px",
                        borderRadius: "6px",
                      }}
                    >
                      RAW RGB INPUT
                    </span>
                  </div>

                  <div
                    style={{
                      borderRadius: "10px",
                      overflow: "hidden",
                      backgroundColor: "#020617",
                      height: "210px",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      border: "1px solid rgba(255, 255, 255, 0.08)",
                    }}
                  >
                    <img
                      src={apiData.intermediate_stages.visualizations.original}
                      alt="Original Input"
                      style={{
                        width: "100%",
                        height: "100%",
                        objectFit: "contain",
                      }}
                    />
                  </div>

                  <div className="telemetry-grid">
                    <div>
                      <span style={{ color: "#64748b" }}>Format:</span> RGB
                    </div>
                    <div>
                      <span style={{ color: "#64748b" }}>Faces:</span>{" "}
                      {apiData.intermediate_stages.metadata.num_faces}
                    </div>
                  </div>
                </div>

                {/* ROW 1 - STAGE 02 (Cyan Accent) */}
                <div
                  className="stage-card stage-2"
                  onClick={() =>
                    setActiveModal({
                      title: "Stage 02: MTCNN Bounding Box & Eye Landmarks",
                      stageName: "STAGE 02",
                      imageSrc:
                        apiData.intermediate_stages.visualizations.bounding_box,
                      accentColor: "#65D0F4",
                      description:
                        "Utilizes Multi-task Cascaded Convolutional Networks (P-Net, R-Net, O-Net) to detect exact facial coordinates and eye keypoint positions.",
                      telemetry: [
                        {
                          label: "BBox [X, Y, W, H]",
                          value: `[${apiData.intermediate_stages.metadata.bbox.join(", ")}]`,
                        },
                        {
                          label: "Left Eye (x, y)",
                          value: `(${apiData.intermediate_stages.metadata.left_eye.join(", ")})`,
                        },
                        {
                          label: "Right Eye (x, y)",
                          value: `(${apiData.intermediate_stages.metadata.right_eye.join(", ")})`,
                        },
                        { label: "Confidence", value: "99.8%" },
                      ],
                    })
                  }
                >
                  <div className="hover-overlay">
                    <span
                      style={{
                        color: "#030712",
                        fontWeight: 700,
                        fontSize: "0.95rem",
                        background: "#65D0F4",
                        padding: "8px 16px",
                        borderRadius: "8px",
                      }}
                    >
                      🔍 Inspect Landmarks
                    </span>
                  </div>

                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      marginBottom: "12px",
                    }}
                  >
                    <span
                      style={{
                        fontSize: "0.85rem",
                        fontWeight: 800,
                        color: "#65D0F4",
                        letterSpacing: "1px",
                      }}
                    >
                      STAGE 02
                    </span>
                    <span
                      style={{
                        fontSize: "0.75rem",
                        color: "#65D0F4",
                        fontWeight: 700,
                        background: "rgba(101, 208, 244, 0.18)",
                        padding: "3px 10px",
                        borderRadius: "6px",
                      }}
                    >
                      MTCNN LANDMARKS
                    </span>
                  </div>

                  <div
                    style={{
                      borderRadius: "10px",
                      overflow: "hidden",
                      backgroundColor: "#020617",
                      height: "210px",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      border: "1px solid rgba(255, 255, 255, 0.08)",
                    }}
                  >
                    <img
                      src={
                        apiData.intermediate_stages.visualizations.bounding_box
                      }
                      alt="MTCNN BBox"
                      style={{
                        width: "100%",
                        height: "100%",
                        objectFit: "contain",
                      }}
                    />
                  </div>

                  <div className="telemetry-grid">
                    <div>
                      <span style={{ color: "#64748b" }}>BBox:</span>{" "}
                      {apiData.intermediate_stages.metadata.bbox
                        .slice(2)
                        .join("x")}
                      px
                    </div>
                    <div>
                      <span style={{ color: "#64748b" }}>Keypoints:</span> 2
                      Eyes
                    </div>
                  </div>
                </div>

                {/* ROW 2 - STAGE 03 (Warm Orange Accent) */}
                <div
                  className="stage-card stage-3"
                  onClick={() =>
                    setActiveModal({
                      title: "Stage 03: Affine Eye Alignment & Cropping",
                      stageName: "STAGE 03",
                      imageSrc:
                        apiData.intermediate_stages.visualizations.aligned_crop,
                      accentColor: "#FF9C4C",
                      description:
                        "Calculates the exact horizontal slope between left and right eyes (arctan2) and performs affine rotation transformation to ensure eyes are 100% horizontal.",
                      telemetry: [
                        {
                          label: "Rotation Angle",
                          value: `${apiData.intermediate_stages.metadata.angle}°`,
                        },
                        {
                          label: "Affine Matrix",
                          value: "cv2.getRotationMatrix2D",
                        },
                        { label: "Interpolation", value: "INTER_CUBIC" },
                        { label: "Orientation", value: "Normalized" },
                      ],
                    })
                  }
                >
                  <div className="hover-overlay">
                    <span
                      style={{
                        color: "#030712",
                        fontWeight: 700,
                        fontSize: "0.95rem",
                        background: "#FF9C4C",
                        padding: "8px 16px",
                        borderRadius: "8px",
                      }}
                    >
                      🔍 Inspect Alignment
                    </span>
                  </div>

                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      marginBottom: "12px",
                    }}
                  >
                    <span
                      style={{
                        fontSize: "0.85rem",
                        fontWeight: 800,
                        color: "#FF9C4C",
                        letterSpacing: "1px",
                      }}
                    >
                      STAGE 03
                    </span>
                    <span
                      style={{
                        fontSize: "0.75rem",
                        color: "#FF9C4C",
                        fontWeight: 700,
                        background: "rgba(255, 156, 76, 0.18)",
                        padding: "3px 10px",
                        borderRadius: "6px",
                      }}
                    >
                      AFFINE ALIGNMENT
                    </span>
                  </div>

                  <div
                    style={{
                      borderRadius: "10px",
                      overflow: "hidden",
                      backgroundColor: "#020617",
                      height: "210px",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      border: "1px solid rgba(255, 255, 255, 0.08)",
                    }}
                  >
                    <img
                      src={
                        apiData.intermediate_stages.visualizations.aligned_crop
                      }
                      alt="Aligned Face"
                      style={{
                        width: "100%",
                        height: "100%",
                        objectFit: "contain",
                      }}
                    />
                  </div>

                  <div className="telemetry-grid">
                    <div>
                      <span style={{ color: "#64748b" }}>Tilt Angle:</span>{" "}
                      {apiData.intermediate_stages.metadata.angle}°
                    </div>
                    <div>
                      <span style={{ color: "#64748b" }}>Crop:</span> Centered
                    </div>
                  </div>
                </div>

                {/* ROW 2 - STAGE 04 (Olive Green Accent) */}
                <div
                  className="stage-card stage-4"
                  onClick={() =>
                    setActiveModal({
                      title:
                        "Stage 04: Standardized Model Input Tensor (224x224)",
                      stageName: "STAGE 04",
                      imageSrc:
                        apiData.intermediate_stages.visualizations.resized_224,
                      accentColor: "#7D9C65",
                      description:
                        "Resizes face crops to standard 224x224 dimensions and normalizes RGB pixel values using ImageNet mean ([0.485, 0.456, 0.406]) and std ([0.229, 0.224, 0.225]).",
                      telemetry: [
                        {
                          label: "Tensor Dimensions",
                          value: "1 x 224 x 224 x 3",
                        },
                        { label: "Data Type", value: "float32" },
                        {
                          label: "Pixel Normalization",
                          value: "[0.0, 1.0] Standardized",
                        },
                        {
                          label: "Model Feed Status",
                          value: "Ready for Inference",
                        },
                      ],
                    })
                  }
                >
                  <div className="hover-overlay">
                    <span
                      style={{
                        color: "#fff",
                        fontWeight: 700,
                        fontSize: "0.95rem",
                        background: "#7D9C65",
                        padding: "8px 16px",
                        borderRadius: "8px",
                      }}
                    >
                      🔍 Inspect Tensor
                    </span>
                  </div>

                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      marginBottom: "12px",
                    }}
                  >
                    <span
                      style={{
                        fontSize: "0.85rem",
                        fontWeight: 800,
                        color: "#7D9C65",
                        letterSpacing: "1px",
                      }}
                    >
                      STAGE 04
                    </span>
                    <span
                      style={{
                        fontSize: "0.75rem",
                        color: "#7D9C65",
                        fontWeight: 700,
                        background: "rgba(125, 156, 101, 0.18)",
                        padding: "3px 10px",
                        borderRadius: "6px",
                      }}
                    >
                      224x224 MODEL TENSOR
                    </span>
                  </div>

                  <div
                    style={{
                      borderRadius: "10px",
                      overflow: "hidden",
                      backgroundColor: "#020617",
                      height: "210px",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      border: "1px solid rgba(255, 255, 255, 0.08)",
                    }}
                  >
                    <img
                      src={
                        apiData.intermediate_stages.visualizations.resized_224
                      }
                      alt="Model Input Tensor"
                      style={{
                        width: "100%",
                        height: "100%",
                        objectFit: "contain",
                      }}
                    />
                  </div>

                  <div className="telemetry-grid">
                    <div>
                      <span style={{ color: "#64748b" }}>Tensor:</span>{" "}
                      (224,224,3)
                    </div>
                    <div>
                      <span style={{ color: "#64748b" }}>Dtype:</span> float32
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* FULL-SCREEN LIGHTBOX MODAL */}
      {activeModal && (
        <div className="modal-backdrop" onClick={() => setActiveModal(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: "20px",
              }}
            >
              <div
                style={{ display: "flex", alignItems: "center", gap: "12px" }}
              >
                <span
                  style={{
                    color: activeModal.accentColor,
                    fontWeight: 900,
                    fontSize: "1.1rem",
                  }}
                >
                  {activeModal.stageName}
                </span>
                <h3
                  style={{
                    margin: 0,
                    fontSize: "1.3rem",
                    fontWeight: 700,
                    color: "#f8fafc",
                  }}
                >
                  {activeModal.title}
                </h3>
              </div>
              <button
                onClick={() => setActiveModal(null)}
                style={{
                  background: "rgba(255, 255, 255, 0.1)",
                  border: "none",
                  color: "#fff",
                  borderRadius: "50%",
                  width: "36px",
                  height: "36px",
                  cursor: "pointer",
                  fontSize: "1.2rem",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                ✕
              </button>
            </div>

            {/* Modal Image Box */}
            <div
              style={{
                borderRadius: "12px",
                overflow: "hidden",
                backgroundColor: "#020617",
                border: `2px solid ${activeModal.accentColor}`,
                padding: "10px",
                textAlign: "center",
                marginBottom: "20px",
              }}
            >
              <img
                src={activeModal.imageSrc}
                alt="Modal Stage High-Res Preview"
                style={{
                  maxHeight: "450px",
                  maxWidth: "100%",
                  objectFit: "contain",
                }}
              />
            </div>

            <p
              style={{
                color: "#cbd5e1",
                lineHeight: 1.6,
                marginBottom: "20px",
              }}
            >
              {activeModal.description}
            </p>

            {/* Full Telemetry Grid */}
            <div
              style={{
                background: "rgba(2, 6, 23, 0.8)",
                borderRadius: "12px",
                padding: "18px",
                border: "1px solid rgba(255,255,255,0.08)",
              }}
            >
              <h4
                style={{
                  margin: "0 0 12px 0",
                  color: activeModal.accentColor,
                  fontSize: "0.9rem",
                  textTransform: "uppercase",
                  letterSpacing: "1px",
                }}
              >
                Technical Execution Metrics
              </h4>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(2, 1fr)",
                  gap: "12px",
                }}
              >
                {activeModal.telemetry.map((t, idx) => (
                  <div key={idx} style={{ fontSize: "0.9rem" }}>
                    <span style={{ color: "#64748b" }}>{t.label}: </span>
                    <strong
                      style={{ color: "#f8fafc", fontFamily: "monospace" }}
                    >
                      {t.value}
                    </strong>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer
        style={{
          borderTop: "1px solid rgba(255, 255, 255, 0.1)",
          padding: "24px",
          textAlign: "center",
          color: "#64748b",
          fontSize: "0.85rem",
          marginTop: "60px",
          background: "rgba(3, 7, 18, 0.7)",
        }}
      >
        Dual-Paradigm Deepfake Detector • Built with React, FastAPI & TensorFlow
      </footer>
    </div>
  );
}
