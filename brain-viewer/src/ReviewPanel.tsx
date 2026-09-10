import { useState } from "react";
import { downloadJson } from "./model";
import type { Prepared } from "./types";

interface Review {
  version: 1;
  fingerprint: string;
  recorded_at: string;
  reviewer: string;
  notes: string;
  planes: string[];
  finding: string;
  scan_id: string;
  reference_id: string | null;
  source_date: string | null;
  reference_date: string | null;
  context: Prepared["context"];
  geometry: Prepared["geometry"];
  provenance: string;
}
export default function ReviewPanel({ prepared }: { prepared: Prepared }) {
  const key = `pie-review-v1:${prepared.fingerprint}`;
  const [records, setRecords] = useState<Review[]>(() => {
    try {
      const a = JSON.parse(localStorage.getItem(key) ?? "[]");
      return Array.isArray(a) ? a : [];
    } catch {
      return [];
    }
  });
  const [reviewer, setReviewer] = useState("");
  const [notes, setNotes] = useState("");
  const [planes, setPlanes] = useState<string[]>([]);
  const [finding, setFinding] = useState("Not assessed");
  const [message, setMessage] = useState("");
  if (!prepared.fingerprint) return null;
  return (
    <section className="control-section review-panel">
      <details>
        <summary>Review notebook · {records.length} entries</summary>
        <p className="empty-note">
          Notes apply only to this acquisition and processing fingerprint. They
          do not change registration status or enable quantitative fusion.
        </p>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            const record: Review = {
              version: 1,
              fingerprint: prepared.fingerprint!,
              recorded_at: new Date().toISOString(),
              reviewer: reviewer.trim(),
              notes: notes.trim(),
              planes,
              finding,
              scan_id: prepared.scan.id,
              reference_id:
                prepared.context?.reference_id ?? prepared.scan.reference_id,
              source_date: prepared.scan.date,
              reference_date: prepared.context?.reference_date ?? null,
              context: prepared.context,
              geometry: prepared.geometry,
              provenance: prepared.scan.provenance,
            };
            if (!record.reviewer || !record.notes) return;
            const next = [...records, record];
            try {
              localStorage.setItem(key, JSON.stringify(next));
              setRecords(next);
              setNotes("");
              setMessage(
                "Saved on this browser. Registration status is unchanged.",
              );
            } catch {
              setMessage(
                "Browser storage is unavailable or full. Export the current notes to keep them.",
              );
              downloadJson(record, `PIE-${prepared.scan.id}-review.json`);
            }
          }}
        >
          <label>
            Reviewer
            <input
              required
              maxLength={120}
              value={reviewer}
              onChange={(e) => setReviewer(e.target.value)}
            />
          </label>
          <fieldset>
            <legend>Planes inspected</legend>
            {["Axial", "Coronal", "Sagittal"].map((p) => (
              <label key={p}>
                <input
                  type="checkbox"
                  checked={planes.includes(p)}
                  onChange={(e) =>
                    setPlanes(
                      e.target.checked
                        ? [...planes, p]
                        : planes.filter((x) => x !== p),
                    )
                  }
                />
                {p}
              </label>
            ))}
          </fieldset>
          <label>
            Visual observation
            <select
              value={finding}
              onChange={(e) => setFinding(e.target.value)}
            >
              <option>Not assessed</option>
              <option>Alignment concern</option>
              <option>Reconstruction concern</option>
              <option>No obvious mismatch noted — visual only</option>
            </select>
          </label>
          <label>
            Review notes
            <textarea
              required
              maxLength={6000}
              rows={3}
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
            />
          </label>
          <button className="button" type="submit">
            Save review note
          </button>
        </form>
        {message && <p role="status">{message}</p>}
        {records.length > 0 && (
          <>
            <button
              className="text-link"
              onClick={() =>
                downloadJson(records, `PIE-${prepared.scan.id}-reviews.json`)
              }
            >
              Export review history
            </button>
            <ol>
              {records
                .slice(-5)
                .reverse()
                .map((r, i) => (
                  <li key={i}>
                    <strong>{r.reviewer}</strong> ·{" "}
                    <time>{new Date(r.recorded_at).toLocaleString()}</time>
                    <p>
                      {r.finding} ·{" "}
                      {r.planes.join(", ") || "No planes recorded"}
                    </p>
                    <p>{r.notes}</p>
                  </li>
                ))}
            </ol>
          </>
        )}
      </details>
    </section>
  );
}
