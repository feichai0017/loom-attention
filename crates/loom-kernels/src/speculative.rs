//! Speculative-decoding verification contracts and CPU references.

use crate::contract::{require_len, ContractError};

/// Contract for deterministic greedy speculative verification.
///
/// Draft tokens are flattened across requests. `cumulative_draft_lengths`
/// stores the inclusive cumulative token count for each request, matching
/// vLLM's speculative-decoding metadata. Each output row has
/// `max_draft_tokens + 1` entries:
///
/// - accepted draft tokens occupy the prefix;
/// - the first target mismatch follows the accepted prefix; or
/// - when every draft token matches, the target-model bonus token follows it.
///
/// Remaining entries are filled with [`PLACEHOLDER_TOKEN_ID`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GreedySpeculativeVerifySpec {
    requests: usize,
    draft_tokens: usize,
    max_draft_tokens: usize,
}

/// Sentinel used for the unused suffix of each speculative output row.
pub const PLACEHOLDER_TOKEN_ID: i32 = -1;

impl GreedySpeculativeVerifySpec {
    /// Creates a validated flattened-ragged verification contract.
    pub fn new(
        requests: usize,
        draft_tokens: usize,
        max_draft_tokens: usize,
    ) -> Result<Self, ContractError> {
        if requests == 0 || draft_tokens == 0 || max_draft_tokens == 0 {
            return Err(ContractError::ZeroDimension);
        }
        let capacity = requests
            .checked_mul(max_draft_tokens)
            .ok_or(ContractError::ElementCountOverflow)?;
        if draft_tokens > capacity {
            return Err(ContractError::DraftTokenCapacityExceeded {
                draft_tokens,
                capacity,
            });
        }
        requests
            .checked_mul(
                max_draft_tokens
                    .checked_add(1)
                    .ok_or(ContractError::ElementCountOverflow)?,
            )
            .ok_or(ContractError::ElementCountOverflow)?;
        Ok(Self {
            requests,
            draft_tokens,
            max_draft_tokens,
        })
    }

    pub const fn requests(self) -> usize {
        self.requests
    }

    pub const fn draft_tokens(self) -> usize {
        self.draft_tokens
    }

    pub const fn max_draft_tokens(self) -> usize {
        self.max_draft_tokens
    }

    pub const fn output_width(self) -> usize {
        self.max_draft_tokens + 1
    }

    pub const fn output_numel(self) -> usize {
        self.requests * self.output_width()
    }
}

/// Verifies greedy draft tokens and compacts each accepted result row.
///
/// The function validates every buffer and every cumulative boundary before
/// mutating outputs.
#[allow(clippy::too_many_arguments)]
pub fn greedy_speculative_verify_reference(
    draft_token_ids: &[i32],
    target_token_ids: &[i64],
    bonus_token_ids: &[i32],
    cumulative_draft_lengths: &[i32],
    output_token_ids: &mut [i32],
    accepted_lengths: &mut [i32],
    emitted_lengths: &mut [i32],
    spec: GreedySpeculativeVerifySpec,
) -> Result<(), ContractError> {
    require_len(
        "draft_token_ids",
        draft_token_ids.len(),
        spec.draft_tokens(),
    )?;
    require_len(
        "target_token_ids",
        target_token_ids.len(),
        spec.draft_tokens(),
    )?;
    require_len("bonus_token_ids", bonus_token_ids.len(), spec.requests())?;
    require_len(
        "cumulative_draft_lengths",
        cumulative_draft_lengths.len(),
        spec.requests(),
    )?;
    require_len(
        "output_token_ids",
        output_token_ids.len(),
        spec.output_numel(),
    )?;
    require_len("accepted_lengths", accepted_lengths.len(), spec.requests())?;
    require_len("emitted_lengths", emitted_lengths.len(), spec.requests())?;

    let mut previous = 0_i32;
    for (request, &current) in cumulative_draft_lengths.iter().enumerate() {
        let segment_length = current.checked_sub(previous);
        if previous < 0
            || current < previous
            || usize::try_from(current).map_or(true, |value| value > spec.draft_tokens())
            || segment_length.is_none_or(|value| {
                usize::try_from(value).map_or(true, |value| value > spec.max_draft_tokens())
            })
        {
            return Err(ContractError::InvalidCumulativeDraftLength {
                request,
                previous,
                current,
                draft_tokens: spec.draft_tokens(),
                max_draft_tokens: spec.max_draft_tokens(),
            });
        }
        previous = current;
    }
    if usize::try_from(previous).ok() != Some(spec.draft_tokens()) {
        return Err(ContractError::FinalCumulativeDraftLengthMismatch {
            expected: spec.draft_tokens(),
            actual: previous,
        });
    }

    let mut target_token_ids_i32 = Vec::with_capacity(spec.draft_tokens());
    for (token, &token_id) in target_token_ids.iter().enumerate() {
        target_token_ids_i32.push(
            i32::try_from(token_id)
                .map_err(|_| ContractError::TargetTokenIdOutOfI32Range { token, token_id })?,
        );
    }

    output_token_ids.fill(PLACEHOLDER_TOKEN_ID);
    let mut start = 0_usize;
    for request in 0..spec.requests() {
        let end = cumulative_draft_lengths[request] as usize;
        let draft_length = end - start;
        let row = &mut output_token_ids
            [request * spec.output_width()..(request + 1) * spec.output_width()];
        let mismatch = (0..draft_length).find(|&position| {
            draft_token_ids[start + position] != target_token_ids_i32[start + position]
        });
        let accepted = mismatch.unwrap_or(draft_length);

        row[..accepted].copy_from_slice(&draft_token_ids[start..start + accepted]);
        row[accepted] = match mismatch {
            Some(position) => target_token_ids_i32[start + position],
            None => bonus_token_ids[request],
        };
        accepted_lengths[request] = accepted as i32;
        emitted_lengths[request] = accepted as i32 + 1;
        start = end;
    }
    Ok(())
}
