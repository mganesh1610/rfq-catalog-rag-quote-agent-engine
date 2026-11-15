// Express + Prisma REST API Controller for CRM Quote Review, Feedback Capture & RBAC Audit Logging
export interface ReviewDecisionRequest {
  quoteId: string;
  actorUserId: string;
  actorRole: "sales_rep" | "quoting_engineer" | "sales_manager";
  decision: "APPROVED" | "REJECTED";
  correctedMpn?: string;
  feedbackNotes?: string;
}

export async function applyQuoteReviewDecision(
  prisma: any,
  req: ReviewDecisionRequest
) {
  if (!["sales_rep", "quoting_engineer", "sales_manager"].includes(req.actorRole)) {
    throw new Error("Unauthorized RBAC role for quote sign-off");
  }
  return prisma.$transaction(async (tx: any) => {
    const updated = await tx.quote.update({
      where: { quoteId: req.quoteId },
      data: { reviewStatus: req.decision },
    });
    await tx.quoteAuditTrail.create({
      data: {
        quoteId: req.quoteId,
        actorUserId: req.actorUserId,
        actorRole: req.actorRole,
        action: `HUMAN_REVIEW_${req.decision}`,
        reviewerFeedback: {
          correctedMpn: req.correctedMpn ?? null,
          notes: req.feedbackNotes ?? "",
        },
      },
    });
    return updated;
  });
}
