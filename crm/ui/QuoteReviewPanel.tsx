import React, { useState } from "react";

export interface QuoteLineItemView {
  extractedMpn: string;
  matchedCatalogMpn: string;
  distributorTier: string;
  quantity: number;
  quotedUnitPrice: number;
  extendedTotal: number;
}

export interface QuoteReviewProps {
  quoteId: string;
  rfqId: string;
  confidenceScore: number;
  reviewStatus: string;
  shipToAddress: string;
  lines: QuoteLineItemView[];
  onSubmitReview: (decision: "APPROVED" | "REJECTED", notes: string) => Promise<void>;
}

export const QuoteReviewPanel: React.FC<QuoteReviewProps> = ({
  quoteId,
  rfqId,
  confidenceScore,
  reviewStatus,
  shipToAddress,
  lines,
  onSubmitReview,
}) => {
  const [notes, setNotes] = useState("");
  const isHighConfidence = confidenceScore >= 0.88;

  return (
    <section aria-label="CRM Quote Human Review Panel">
      <header>
        <h2>{quoteId} — Source {rfqId}</h2>
        <span>
          Confidence: {(confidenceScore * 100).toFixed(1)}% ({reviewStatus})
        </span>
        <p>Ship To: {shipToAddress}</p>
      </header>
      {!isHighConfidence && (
        <div role="alert">
          Low-confidence OCR or distributor MPN match detected; verify part numbers before approval.
        </div>
      )}
      <table>
        <thead>
          <tr>
            <th>Extracted MPN</th>
            <th>Matched Catalog MPN</th>
            <th>Distributor</th>
            <th>Qty</th>
            <th>Unit Quote ($)</th>
            <th>Extended ($)</th>
          </tr>
        </thead>
        <tbody>
          {lines.map((ln, idx) => (
            <tr key={`${ln.matchedCatalogMpn}-${idx}`}>
              <td>{ln.extractedMpn}</td>
              <td>{ln.matchedCatalogMpn}</td>
              <td>{ln.distributorTier}</td>
              <td>{ln.quantity}</td>
              <td>{ln.quotedUnitPrice.toFixed(2)}</td>
              <td>{ln.extendedTotal.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <textarea
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        placeholder="Capture MPN override or distributor pricing feedback for retraining..."
      />
      <div>
        <button type="button" onClick={() => onSubmitReview("APPROVED", notes)}>
          Approve Quote
        </button>
        <button type="button" onClick={() => onSubmitReview("REJECTED", notes)}>
          Escalate / Reject
        </button>
      </div>
    </section>
  );
};
